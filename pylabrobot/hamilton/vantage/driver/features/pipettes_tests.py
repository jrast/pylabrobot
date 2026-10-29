import re
import unittest
from typing import Any, List, Optional, Tuple

from pylabrobot.hamilton.vantage.driver.errors import VantageFirmwareError
from pylabrobot.hamilton.vantage.driver.features.pipettes import (
  Pipettes,
  PipettesConfiguration,
  _get_dispense_mode,
)
from pylabrobot.hamilton.vantage.driver.master import VantageDriver
from pylabrobot.hamilton.vantage.driver.master_tests import ScriptedIO, echo, machine
from pylabrobot.legacy.liquid_handling.backends.hamilton.vantage_backend import VantageBackend
from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.errors import HasTipError, NoTipError
from pylabrobot.resources.hamilton.tip_carriers import hamilton_tip_carrier_L5
from pylabrobot.resources.hamilton.tip_creators import HamiltonTip, TipPickupMethod
from pylabrobot.resources.hamilton.tip_racks import hamilton_96_tiprack_300uL
from pylabrobot.resources.hamilton.vantage_decks import VantageDeck
from pylabrobot.resources.tip_rack import TipRack
from pylabrobot.resources.tip_tracking import set_tip_tracking

PIPETTES_LOGGER = "pylabrobot.hamilton.vantage.driver.features.pipettes"


def without_id(command: str) -> str:
  """A command without the id it was sent with."""
  return re.sub(r"id\d+", "", command, count=1)


class LegacyCatcher(VantageBackend):
  """The legacy backend, writing what it would send into a list instead of to a machine."""

  def __init__(self, num_channels: int):
    super().__init__()
    self.commands: List[str] = []
    self._num_channels = num_channels

  async def send_command(  # type: ignore[override]
    self,
    module: str,
    command: str,
    auto_id: bool = True,
    tip_pattern: Optional[List[bool]] = None,
    write_timeout: Optional[int] = None,
    read_timeout: Optional[int] = None,
    wait: bool = True,
    fmt: Optional[Any] = None,
    **kwargs,
  ):
    cmd, _ = self._assemble_command(
      module=module, command=command, auto_id=auto_id, tip_pattern=tip_pattern, **kwargs
    )
    self.commands.append(cmd)


class PipettesTestCase(unittest.IsolatedAsyncioTestCase):
  """A driver on a machine with `num_channels` channels, discovered."""

  num_channels = 8

  async def asyncSetUp(self):
    self.io = ScriptedIO(machine(num_channels=self.num_channels))
    self.driver = VantageDriver(self.io, read_timeout=5)
    self.driver._connected = True
    self.driver._router.start()
    await self.driver.discover()
    self.io.written.clear()
    pipettes = self.driver.pipettes
    assert pipettes is not None
    self.pipettes: Pipettes = pipettes

  async def asyncTearDown(self):
    self.driver._router.stop()


class TestConfiguration(unittest.TestCase):
  """Converting a value in mm to what the wire carries."""

  def test_a_value_is_written_in_tenths_of_a_millimeter(self):
    configuration = PipettesConfiguration()
    self.assertEqual(configuration.to_increments(245.0), 2450)
    self.assertEqual(configuration.to_increments(123.5), 1235)
    self.assertEqual(configuration.to_increments(0.0), 0)
    self.assertEqual(configuration.to_increments(-7.5), -75)

  def test_a_value_finer_than_the_wire_goes_to_the_nearest_tenth(self):
    configuration = PipettesConfiguration()
    self.assertEqual(configuration.to_increments(100.26), 1003)
    self.assertEqual(configuration.to_increments(100.24), 1002)
    self.assertEqual(configuration.to_increments(-100.26), -1003)

  def test_a_value_with_float_error_is_rounded_to_the_tenth_it_means(self):
    self.assertEqual(PipettesConfiguration().to_increments(389.1), 3891)
    self.assertEqual(PipettesConfiguration().to_increments(201.6), 2016)


class TestQueries(PipettesTestCase):
  """What the channels are asked."""

  async def test_the_tips_come_back_one_for_each_channel_as_the_machine_reports_them(self):
    self.io.answers = machine(tips=[True, False, False, True, False, False, False, False])
    tips = await self.pipettes.sense_tip_presence()
    self.assertEqual(tips, [1, 0, 0, 1, 0, 0, 0, 0])
    self.assertTrue(all(type(tip) is int for tip in tips))
    self.assertEqual([without_id(command) for command in self.io.written], ["A1PMQA"])

  async def test_the_initialization_status_is_true_only_for_1(self):
    self.io.answers = machine(channels_initialized=True)
    self.assertTrue(await self.pipettes.request_initialization_status())
    self.io.answers = machine(channels_initialized=False)
    self.assertFalse(await self.pipettes.request_initialization_status())
    self.assertEqual([command[:6] for command in self.io.written], ["A1PMQW", "A1PMQW"])


class TestInitialize(PipettesTestCase):
  """Initializing the channels, which moves them."""

  async def test_by_default_the_wire_carries_what_the_legacy_backend_sends_on_setup(self):
    legacy = LegacyCatcher(self.num_channels)
    await legacy.pip_initialize(
      x_position=[7095] * 8,
      y_position=[3891, 3623, 3355, 3087, 2819, 2551, 2283, 2016],
      begin_z_deposit_position=[2450] * 8,
      end_z_deposit_position=[1235] * 8,
      minimal_height_at_command_end=[2450] * 8,
      tip_pattern=[True] * 8,
      tip_type=[1] * 8,
      TODO_DI_2=70,
    )
    await self.pipettes.initialize(tip_type=[1] * 8)
    self.assertEqual(without_id(self.io.written[0]), without_id(legacy.commands[0]))

  async def test_the_defaults_for_eight_channels(self):
    await self.pipettes.initialize()
    self.assertEqual(
      without_id(self.io.written[0]),
      "A1PMDI"
      "xp"
      + " ".join(["7095"] * 8)
      + "yp3891 3623 3355 3087 2819 2551 2283 2016"
      + "tp"
      + " ".join(["2450"] * 8)
      + "tz"
      + " ".join(["1235"] * 8)
      + "te"
      + " ".join(["2450"] * 8)
      + "tm"
      + " ".join(["1"] * 8)
      + "tt"
      + " ".join(["4"] * 8)
      + "ts70",
    )

  async def test_the_command_goes_to_the_channels_and_is_given_the_time_they_need(self):
    seen = {}
    send = self.driver._router.send

    async def spying(cmd, id_, write_timeout=None, read_timeout=None, wait=True):
      seen[cmd[:6]] = read_timeout
      return await send(cmd, id_, write_timeout, read_timeout, wait)

    self.driver._router.send = spying  # type: ignore[method-assign]
    await self.pipettes.initialize()
    self.assertEqual(seen, {"A1PMDI": 300})

  async def test_what_is_given_is_taken_in_mm_and_converted(self):
    await self.pipettes.initialize(
      x_position=[100.5] * 8,
      y_positions=[10.0 * (i + 1) for i in range(8)],
      begin_of_tip_deposit=[200.0] * 8,
      end_of_tip_deposit=[100.3] * 8,
      minimal_height_at_command_end=[300.0] * 8,
      tip_type=[0, 1, 2, 3, 4, 5, 6, 7],
      x_shift_during_discard=-2.5,
    )
    command = without_id(self.io.written[0])
    self.assertIn("xp" + " ".join(["1005"] * 8), command)
    self.assertIn("yp100 200 300 400 500 600 700 800", command)
    self.assertIn("tp" + " ".join(["2000"] * 8), command)
    self.assertIn("tz" + " ".join(["1003"] * 8), command)
    self.assertIn("te" + " ".join(["3000"] * 8), command)
    self.assertIn("tt0 1 2 3 4 5 6 7", command)
    self.assertTrue(command.endswith("ts-25"), command)

  async def test_channels_can_be_left_out(self):
    await self.pipettes.initialize(tip_pattern=[True] + [False] * 7)
    self.assertIn("tm1 0 0 0 0 0 0 0", without_id(self.io.written[0]))

  async def test_a_list_that_does_not_hold_one_value_for_each_channel_is_refused(self):
    for name in (
      "x_position",
      "y_positions",
      "begin_of_tip_deposit",
      "end_of_tip_deposit",
      "minimal_height_at_command_end",
      "tip_pattern",
      "tip_type",
    ):
      with self.assertRaises(ValueError, msg=name) as caught:
        await self.pipettes.initialize(**{name: [1] * 7})  # type: ignore[arg-type]
      self.assertIn(name, str(caught.exception))
    self.assertEqual(self.io.written, [])

  async def test_a_value_out_of_range_is_refused_and_nothing_is_sent(self):
    refused = (
      {"x_position": [-0.1] * 8},
      {"x_position": [5000.1] * 8},
      {"y_positions": [-0.1] * 8},
      {"y_positions": [650.1] * 8},
      {"begin_of_tip_deposit": [360.1] * 8},
      {"end_of_tip_deposit": [-1.0] * 8},
      {"minimal_height_at_command_end": [361.0] * 8},
      {"tip_type": [200] * 8},
      {"tip_type": [-1] * 8},
      {"x_shift_during_discard": 100.1},
      {"x_shift_during_discard": -100.1},
    )
    for arguments in refused:
      with self.assertRaises(ValueError, msg=str(arguments)):
        await self.pipettes.initialize(**arguments)
    self.assertEqual(self.io.written, [])

  async def test_the_ends_of_every_range_are_accepted(self):
    await self.pipettes.initialize(
      x_position=[0.0] * 4 + [5000.0] * 4,
      y_positions=[0.0] * 4 + [650.0] * 4,
      begin_of_tip_deposit=[0.0] * 4 + [360.0] * 4,
      tip_type=[0] * 4 + [199] * 4,
      x_shift_during_discard=100.0,
    )
    await self.pipettes.initialize(x_shift_during_discard=-100.0)
    self.assertEqual(len(self.io.written), 2)

  async def test_a_reported_error_reaches_the_caller(self):
    self.io.answers = lambda command: [echo(command, 'es"P170"', error=99)]
    with self.assertRaises(VantageFirmwareError):
      await self.pipettes.initialize()


class TestAMachineWithAnotherChannelCount(PipettesTestCase):
  num_channels = 4

  async def test_there_is_no_default_discard_position_for_it(self):
    with self.assertRaises(ValueError) as caught:
      await self.pipettes.initialize()
    self.assertIn("y_positions", str(caught.exception))
    self.assertIn("4 channels", str(caught.exception))
    self.assertEqual(self.io.written, [])

  async def test_it_initializes_where_it_is_told_to(self):
    await self.pipettes.initialize(y_positions=[400.0, 350.0, 300.0, 250.0])
    command = without_id(self.io.written[0])
    self.assertIn("yp4000 3500 3000 2500", command)
    self.assertIn("xp7095 7095 7095 7095", command)
    self.assertIn("tm1 1 1 1", command)
    self.assertNotIn("&", command)

  async def test_the_defaults_are_a_public_table_that_can_be_extended(self):
    self.pipettes.default_y_positions = {4: [400.0, 350.0, 300.0, 250.0]}
    await self.pipettes.initialize()
    self.assertIn("yp4000 3500 3000 2500", self.io.written[0])


class TestBeforeDiscovery(unittest.TestCase):
  def test_the_number_of_channels_is_not_known(self):
    pipettes = Pipettes(VantageDriver(ScriptedIO()))
    with self.assertRaises(RuntimeError):
      pipettes.num_channels  # noqa: B018


if __name__ == "__main__":
  unittest.main()


def deck_with_tips(track: int = 15, site: int = 1) -> Tuple[VantageDeck, TipRack]:
  """A 1.3 m deck with a tip carrier on a track, and a rack of 300 uL tips in one of its sites.

  The rack is put into its site without a location: a holder centres a rack over the opening it
  sinks into, and sinks it by the rack's own sinking depth. Placing it at the holder's corner
  instead would put it about 4 mm off in X and Y, and 6 mm too high.
  """
  deck = VantageDeck(size=1.3, name="deck")
  carrier = hamilton_tip_carrier_L5("tip_carrier")
  deck.assign_child_resource(carrier, track=track)
  rack = hamilton_96_tiprack_300uL("rack")
  carrier[site].assign_child_resource(rack)
  return deck, rack


class TipsTestCase(unittest.IsolatedAsyncioTestCase):
  """A driver with a deck, a rack of 300 uL tips, and the machine discovered."""

  async def asyncSetUp(self):
    self.deck, self.rack = deck_with_tips()
    self.io = ScriptedIO(machine())
    self.driver = VantageDriver(self.io, deck=self.deck, read_timeout=5)
    self.driver._connected = True
    self.driver._router.start()
    await self.driver.discover()
    self.io.written.clear()
    pipettes = self.driver.pipettes
    assert pipettes is not None
    self.pipettes: Pipettes = pipettes
    self.spots = self.rack.get_items("A1:H1")

  async def asyncTearDown(self):
    self.driver._router.stop()

  def sent(self) -> List[str]:
    return [without_id(command) for command in self.io.written]

  def command(self, module_and_command: str) -> str:
    """The one command with that module and command."""
    found = [command for command in self.sent() if command.startswith(module_and_command)]
    self.assertEqual(len(found), 1, self.sent())
    return found[0]

  def parameter(self, command: str, key: str) -> str:
    """The value of one parameter of a command, as it was written."""
    found = re.search(rf"{key}([^a-z]+)", command[6:])
    assert found is not None, (key, command)
    return found.group(1).strip()


class TestPickUpTips(TipsTestCase):
  """Picking tips up, which moves the channels."""

  def where(self) -> Tuple[List[int], List[int], int, int]:
    """The wire positions of the spots, and where the pickup begins and ends, in 0.1 mm."""
    config = PipettesConfiguration()
    xs: List[int] = []
    ys: List[int] = []
    for spot in self.spots:
      location = spot.get_location_wrt(self.deck, x="c", y="c", z="b")
      xs.append(config.to_increments(location.x))
      ys.append(config.to_increments(location.y))
    z = self.spots[0].get_location_wrt(self.deck, z="b").z
    return xs, ys, config.to_increments(z + 8.0), config.to_increments(z)

  async def test_the_heights_are_the_ones_legacy_was_tested_with(self):
    # The same deck, carrier, rack and tips as the legacy tests: a 300 uL tip in its rack.
    await self.pipettes.pick_up_tips(self.spots[:1], use_channels=[0])
    command = self.command("A1PMTP")
    self.assertEqual(self.parameter(command, "tp").split(), ["2244"] + ["0"] * 7)
    self.assertEqual(self.parameter(command, "tz").split(), ["2164"] + ["0"] * 7)

  async def test_the_wire_is_what_legacy_sends(self):
    xs, ys, begin, end = self.where()
    legacy = LegacyCatcher(self.pipettes.num_channels)
    await legacy.pip_tip_pick_up(
      x_position=xs,
      y_position=ys,
      tip_pattern=[True] * 8,
      tip_type=[0] * 8,
      begin_z_deposit_position=[begin] * 8,
      end_z_deposit_position=[end] * 8,
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
      minimal_height_at_command_end=[2450] * 8,
      blow_out_air_volume=[0] * 8,
      tip_handling_method=[1] * 8,
    )
    await self.pipettes.pick_up_tips(self.spots)
    for key in ("xp", "yp", "tm", "tt", "tp", "tz", "th", "te", "ba", "td"):
      self.assertEqual(
        self.parameter(self.command("A1PMTP"), key),
        self.parameter(legacy.commands[0], key),
        key,
      )

  async def test_a_plain_300_ul_tip_needs_no_tip_type_written(self):
    await self.pipettes.pick_up_tips(self.spots)
    self.assertEqual([command[:6] for command in self.sent()], ["A1PMTP"])
    self.assertEqual(self.parameter(self.command("A1PMTP"), "tt"), " ".join(["0"] * 8))

  async def test_the_entry_is_remembered_between_commands(self):
    await self.pipettes.pick_up_tips(self.spots)
    await self.pipettes.drop_tips(self.spots)
    self.io.written.clear()
    await self.pipettes.pick_up_tips(self.spots)
    self.assertEqual([command[:6] for command in self.sent()], ["A1PMTP"])

  async def test_the_pickup_ends_at_the_spot_and_begins_a_collar_above_it(self):
    _, _, begin, end = self.where()
    await self.pipettes.pick_up_tips(self.spots)
    command = self.command("A1PMTP")
    self.assertEqual(self.parameter(command, "tz"), " ".join([str(end)] * 8))
    self.assertEqual(self.parameter(command, "tp"), " ".join([str(begin)] * 8))
    self.assertEqual(self.parameter(command, "th"), " ".join(["2450"] * 8))
    self.assertEqual(self.parameter(command, "te"), " ".join(["2450"] * 8))
    self.assertEqual(self.parameter(command, "td"), " ".join(["1"] * 8))
    self.assertEqual(self.parameter(command, "ba"), " ".join(["0"] * 8))

  async def test_where_the_pickup_begins_and_ends_can_be_given(self):
    await self.pipettes.pick_up_tips(
      self.spots,
      begin_tip_pick_up_process=240.0,
      end_tip_pick_up_process=230.0,
      minimum_traverse_height_start=300.0,
    )
    command = self.command("A1PMTP")
    self.assertEqual(self.parameter(command, "tp"), " ".join(["2400"] * 8))
    self.assertEqual(self.parameter(command, "tz"), " ".join(["2300"] * 8))
    self.assertEqual(self.parameter(command, "th"), " ".join(["3000"] * 8))
    self.assertEqual(self.parameter(command, "te"), " ".join(["3000"] * 8))

  async def test_a_traverse_height_at_or_below_the_safety_height_is_refused(self):
    with self.assertRaises(ValueError):
      await self.pipettes.pick_up_tips(self.spots, minimum_traverse_height_start=200.0)
    self.assertEqual(self.io.written, [])

  async def test_a_tip_collected_out_of_wash_liquid_says_so(self):
    set_tip_tracking(True)
    try:
      for spot in self.spots:
        assert isinstance(spot.tip, HamiltonTip)
        spot.tip.pickup_method = TipPickupMethod.OUT_OF_WASH_LIQUID
      await self.pipettes.pick_up_tips(self.spots)
    finally:
      set_tip_tracking(False)
    self.assertEqual(self.parameter(self.command("A1PMTP"), "td"), " ".join(["2"] * 8))

  async def test_the_spots_are_where_the_tips_are(self):
    xs, ys, _, _ = self.where()
    await self.pipettes.pick_up_tips(self.spots)
    command = self.command("A1PMTP")
    self.assertEqual(self.parameter(command, "xp"), " ".join(str(x) for x in xs))
    self.assertEqual(self.parameter(command, "yp"), " ".join(str(y) for y in ys))

  async def test_the_tips_leave_their_spots_and_get_onto_their_channels(self):
    await self.pipettes.pick_up_tips(self.spots)
    for channel, spot in enumerate(self.spots):
      self.assertIsNone(spot.tip, spot.name)
      mounted = self.pipettes.get_mounted_tip(channel)
      self.assertIsNotNone(mounted, channel)
      self.assertEqual(mounted.name.split("#")[0], spot.name)  # type: ignore[union-attr]

  async def test_a_channel_that_already_carries_a_tip_is_refused(self):
    await self.pipettes.pick_up_tips(self.spots)
    self.io.written.clear()
    with self.assertRaises(HasTipError):
      await self.pipettes.pick_up_tips(self.spots)
    self.assertEqual(self.io.written, [])

  async def test_a_spot_that_holds_no_tip_is_refused_while_tips_are_tracked(self):
    set_tip_tracking(True)
    try:
      self.spots[0].unassign_tip()
      with self.assertRaises(NoTipError):
        await self.pipettes.pick_up_tips(self.spots)
    finally:
      set_tip_tracking(False)
    self.assertEqual(self.io.written, [])

  async def test_channels_can_be_given_and_are_ascending(self):
    await self.pipettes.pick_up_tips(self.spots[:2], use_channels=[0, 1])
    self.assertIsNotNone(self.pipettes.get_mounted_tip(0))
    self.assertIsNotNone(self.pipettes.get_mounted_tip(1))
    self.assertIsNone(self.pipettes.get_mounted_tip(2))

  async def test_the_channels_that_take_no_part_are_written_out_as_zero(self):
    # A `&` after a short list means "the last value again", so it would take every remaining
    # channel to the last position and pick up a tip there.
    xs, ys, _, _ = self.where()
    await self.pipettes.pick_up_tips(self.spots[:2], use_channels=[0, 1])
    command = self.command("A1PMTP")
    self.assertNotIn("&", command)
    self.assertEqual(self.parameter(command, "tm"), "1 1 0 0 0 0 0 0")
    self.assertEqual(self.parameter(command, "xp"), " ".join(str(x) for x in xs[:2] + [0] * 6))
    self.assertEqual(self.parameter(command, "yp"), " ".join(str(y) for y in ys[:2] + [0] * 6))

  async def test_channels_that_are_not_ascending_are_refused(self):
    with self.assertRaises(ValueError):
      await self.pipettes.pick_up_tips(self.spots[:2], use_channels=[1, 0])

  async def test_two_tips_closer_than_the_channels_stand_are_taken_but_warned_about(self):
    with self.assertLogs(PIPETTES_LOGGER, "WARNING") as logs:
      await self.pipettes.pick_up_tips(
        self.spots[:2], use_channels=[0, 1], offsets=[Coordinate(0, 0, 0), Coordinate(0, 0.5, 0)]
      )
    self.assertIn("8.5 mm", logs.output[0])
    self.assertIn("9.0 mm", logs.output[0])
    self.assertIsNotNone(self.pipettes.get_mounted_tip(1))

  async def test_a_tenth_below_the_minimum_is_already_too_close(self):
    with self.assertLogs(PIPETTES_LOGGER, "WARNING") as logs:
      await self.pipettes.pick_up_tips(
        self.spots[:2], use_channels=[0, 1], offsets=[Coordinate(0, 0, 0), Coordinate(0, 0.1, 0)]
      )
    self.assertIn("8.9 mm", logs.output[0])

  async def test_tips_at_the_minimum_spacing_are_taken_without_a_word(self):
    with self.assertNoLogs(PIPETTES_LOGGER, "WARNING"):
      await self.pipettes.pick_up_tips(self.spots[:2], use_channels=[0, 1])
    self.assertIsNotNone(self.pipettes.get_mounted_tip(1))

  async def test_tips_a_column_apart_may_be_any_distance_in_y_without_a_word(self):
    with self.assertNoLogs(PIPETTES_LOGGER, "WARNING"):
      await self.pipettes.pick_up_tips(
        self.spots[:2], use_channels=[0, 1], offsets=[Coordinate(0, 0, 0), Coordinate(9.0, 1.0, 0)]
      )
    self.assertIsNotNone(self.pipettes.get_mounted_tip(1))

  async def test_a_position_the_channels_cannot_reach_is_refused(self):
    with self.assertRaises(ValueError):
      await self.pipettes.pick_up_tips(self.spots, offsets=[Coordinate(6000.0, 0.0, 0.0)] * 8)
    self.assertEqual(self.io.written, [])

  async def test_a_failed_pickup_takes_the_machines_word_for_what_came_up(self):
    def answers(command: str) -> Optional[List[str]]:
      if command.startswith("A1PMTP"):
        return [echo(command, 'es"P123"', error=1)]
      if command.startswith("A1PMQA"):
        return [echo(command, "rt1 0 0 0 0 0 0 0")]
      return machine()(command)

    self.io.answers = answers
    with self.assertRaises(VantageFirmwareError):
      await self.pipettes.pick_up_tips(self.spots)
    self.assertIsNotNone(self.pipettes.get_mounted_tip(0))
    self.assertIsNone(self.spots[0].tip)
    self.assertIsNone(self.pipettes.get_mounted_tip(1))
    self.assertIsNotNone(self.spots[1].tip)


class TestDropTips(TipsTestCase):
  """Putting tips back into their spots."""

  async def asyncSetUp(self):
    await super().asyncSetUp()
    await self.pipettes.pick_up_tips(self.spots)
    self.io.written.clear()

  def where(self) -> Tuple[List[int], List[int]]:
    config = PipettesConfiguration()
    xs: List[int] = []
    ys: List[int] = []
    for spot in self.spots:
      location = spot.get_location_wrt(self.deck, x="c", y="c", z="b")
      xs.append(config.to_increments(location.x))
      ys.append(config.to_increments(location.y))
    return xs, ys

  def end(self) -> int:
    """Where the tip's end sits in its spot, in 0.1 mm."""
    seat = self.spots[0].get_location_wrt(self.deck, z="b").z
    return PipettesConfiguration().to_increments(seat - (59.9 - 8.0))

  async def test_the_heights_are_the_ones_legacy_was_tested_with(self):
    await self.pipettes.drop_tips(self.spots[:1], use_channels=[0])
    command = self.command("A1PMTR")
    self.assertEqual(self.parameter(command, "tp").split(), ["1745"] + ["0"] * 7)
    self.assertEqual(self.parameter(command, "tz").split(), ["1645"] + ["0"] * 7)

  async def test_the_wire_is_what_legacy_sends(self):
    xs, ys = self.where()
    end = self.end()
    legacy = LegacyCatcher(self.pipettes.num_channels)
    await legacy.pip_tip_discard(
      x_position=xs,
      y_position=ys,
      begin_z_deposit_position=[end + 100] * 8,
      end_z_deposit_position=[end] * 8,
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
      minimal_height_at_command_end=[2450] * 8,
      tip_pattern=[True] * 8,
      TODO_TR_2=0,
      tip_handling_method=[0] * 8,
    )
    await self.pipettes.drop_tips(self.spots)
    for key in ("xp", "yp", "tp", "tz", "th", "te", "tm", "ts", "td"):
      self.assertEqual(
        self.parameter(self.command("A1PMTR"), key),
        self.parameter(legacy.commands[0], key),
        key,
      )

  async def test_the_tips_go_back_into_their_spots(self):
    await self.pipettes.drop_tips(self.spots)
    for channel, spot in enumerate(self.spots):
      self.assertIsNotNone(spot.tip, spot.name)
      self.assertIsNone(self.pipettes.get_mounted_tip(channel))
    self.assertEqual([command[:6] for command in self.sent()], ["A1PMTR"])

  async def test_a_channel_without_a_tip_is_refused(self):
    await self.pipettes.drop_tips(self.spots[:1], use_channels=[0])
    with self.assertRaises(NoTipError):
      await self.pipettes.drop_tips(self.spots[:1], use_channels=[0])

  async def test_a_spot_that_holds_a_tip_already_is_refused(self):
    await self.pipettes.drop_tips(self.spots[:1], use_channels=[0])
    self.io.written.clear()
    with self.assertRaises(HasTipError):
      await self.pipettes.drop_tips(self.spots[:1], use_channels=[1])
    self.assertEqual(self.io.written, [])


class PositioningTestCase(PipettesTestCase):
  """A driver on a machine with eight channels, discovered, sending no ids in its tests."""

  def sent(self) -> List[str]:
    return [without_id(command) for command in self.io.written]


class TestPositioningMoves(PositioningTestCase):
  """Moving the channels, all of them or one at a time."""

  async def test_every_channel_moves_along_y(self):
    await self.pipettes.move_to_y_positions({channel: 100.0 + channel for channel in range(8)})
    self.assertEqual(self.sent(), ["A1PMDYyp1000 1010 1020 1030 1040 1050 1060 1070"])

  async def test_every_channel_move_matches_the_legacy_command(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.position_all_channels_in_y_direction(y_position=[100, 200, 300, 400, 500])
    await self.pipettes.move_to_y_positions(
      {0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0, 4: 50.0, 5: 60.0, 6: 70.0, 7: 80.0}
    )
    self.assertEqual(self.sent()[-1][:6], "A1PMDY")
    self.assertEqual(self.sent()[-1].split("yp")[1].split()[0], "100")
    self.assertEqual(without_id(catcher.commands[-1])[:6], "A1PMDY")

  async def test_every_channel_moves_along_z(self):
    await self.pipettes.move_to_z_positions({channel: 10.0 * channel for channel in range(8)})
    self.assertEqual(self.sent(), ["A1PMDZzp0 100 200 300 400 500 600 700"])

  async def test_one_channel_moves_along_y(self):
    await self.pipettes.move_to_y_position(channel=2, y=123.4)
    self.assertEqual(self.sent(), ["A1PMDVpn3yj1234"])

  async def test_one_channel_move_matches_the_legacy_command(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.position_single_channel_in_y_direction(channel_index=3, y_position=1234)
    await self.pipettes.move_to_y_position(channel=2, y=123.4)
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_one_channel_moves_along_z(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.position_single_channel_in_z_direction(channel_index=5, z_position=2500)
    await self.pipettes.move_to_z_position(channel=4, z=250.0)
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))
    self.assertEqual(self.sent()[-1], "A1PMDUpn5zj2500")

  async def test_a_place_for_each_channel_of_the_pattern(self):
    await self.pipettes.move_to_positions(
      {1: 100.0, 3: 200.0}, {1: 300.0, 3: 400.0}, {1: 50.0, 3: 60.0}
    )
    self.assertEqual(
      self.sent(),
      [
        "A1PMDNtm0 1 0 1 0 0 0 0xp0 1000 0 2000 0 0 0 0yp0 3000 0 4000 0 0 0 0"
        "th2450 2450 2450 2450 2450 2450 2450 2450zp0 500 0 600 0 0 0 0"
      ],
    )

  async def test_a_place_matches_the_legacy_command(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.move_to_defined_position(
      x_position=[0, 1000, 0, 2000, 0, 0, 0, 0],
      y_position=[0, 3000, 0, 4000, 0, 0, 0, 0],
      tip_pattern=[False, True, False, True, False, False, False, False],
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
      z_position=[0, 500, 0, 600, 0, 0, 0, 0],
    )
    await self.pipettes.move_to_positions(
      {1: 100.0, 3: 200.0}, {1: 300.0, 3: 400.0}, {1: 50.0, 3: 60.0}
    )
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_the_travel_height_can_be_given(self):
    await self.pipettes.move_to_positions(
      {0: 1.0}, {0: 2.0}, {0: 3.0}, minimal_traverse_height=300.0
    )
    self.assertIn("th3000", self.sent()[-1])

  async def test_the_axes_must_name_the_same_channels(self):
    with self.assertRaises(ValueError):
      await self.pipettes.move_to_positions({0: 1.0}, {0: 2.0, 1: 3.0}, {0: 4.0})
    self.assertEqual(self.sent(), [])

  async def test_nothing_is_moved_when_no_channel_is_named(self):
    with self.assertRaises(ValueError):
      await self.pipettes.move_to_positions({}, {}, {})
    self.assertEqual(self.sent(), [])

  async def test_a_channel_is_exposed(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.expose_channel_n(channel_index=4)
    await self.pipettes.expose_channel(channel=3)
    self.assertEqual(self.sent(), ["A1PMDQpn4"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_a_channel_the_machine_has_not_is_refused(self):
    for move in (
      lambda: self.pipettes.move_to_y_position(channel=8, y=10.0),
      lambda: self.pipettes.move_to_z_position(channel=-1, z=10.0),
      lambda: self.pipettes.expose_channel(channel=8),
    ):
      with self.assertRaises(ValueError):
        await move()
    self.assertEqual(self.sent(), [])

  async def test_a_position_out_of_reach_is_refused(self):
    refused = [
      lambda: self.pipettes.move_to_y_positions({channel: 650.1 for channel in range(8)}),
      lambda: self.pipettes.move_to_z_positions({channel: -0.1 for channel in range(8)}),
      lambda: self.pipettes.move_to_positions({0: 5000.1}, {0: 1.0}, {0: 1.0}),
      lambda: self.pipettes.move_to_positions({0: 1.0}, {0: 650.1}, {0: 1.0}),
      lambda: self.pipettes.move_to_positions({0: 1.0}, {0: 1.0}, {0: 360.1}),
    ]
    for move in refused:
      with self.assertRaises(ValueError):
        await move()
    self.assertEqual(self.sent(), [])

  async def test_a_travel_height_below_the_deck_is_refused(self):
    with self.assertRaises(ValueError):
      await self.pipettes.move_to_positions(
        {0: 1.0}, {0: 1.0}, {0: 1.0}, minimal_traverse_height=200.0
      )
    self.assertEqual(self.sent(), [])

  async def test_every_channel_of_an_all_channel_move_is_required(self):
    with self.assertRaises(ValueError) as caught:
      await self.pipettes.move_to_y_positions({0: 10.0})
    self.assertIn("[1, 2, 3, 4, 5, 6, 7]", str(caught.exception))
    with self.assertRaises(ValueError):
      await self.pipettes.move_to_z_positions({channel: 10.0 for channel in range(8)} | {8: 1.0})
    self.assertEqual(self.sent(), [])


class TestPositionQueries(PositioningTestCase):
  """Where the channels stand, whose replies are not read yet."""

  async def test_y_positions(self):
    self.io.answers = lambda command: [echo(command, "yp100 200")]
    reply = await self.pipettes.request_y_positions()
    self.assertEqual(self.sent(), ["A1PMRY"])
    self.assertIn("er0yp100 200", reply)

  async def test_y_position_of_one_channel(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.request_y_position_of_channel_n(channel_index=3)
    reply = await self.pipettes.request_y_position(channel=2)
    self.assertEqual(self.sent(), ["A1PMRBpn3"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))
    self.assertIn("A1PMRBid", reply)

  async def test_z_positions(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.request_z_positions_of_all_channels()
    await self.pipettes.request_z_positions()
    self.assertEqual(self.sent(), ["A1PMRZ"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_z_position_of_one_channel(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.request_z_position_of_channel_n(channel_index=5)
    await self.pipettes.request_z_position(channel=4)
    self.assertEqual(self.sent(), ["A1PMRDpn5"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_the_height_of_the_last_lld(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.request_height_of_last_lld()
    await self.pipettes.request_height_of_last_lld()
    self.assertEqual(self.sent(), ["A1PMRL"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_the_dispense_on_fly_status(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.request_channel_dispense_on_fly_status()
    await self.pipettes.request_dispense_on_fly_status()
    self.assertEqual(self.sent(), ["A1PMQF"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_a_channel_the_machine_has_not_is_refused(self):
    with self.assertRaises(ValueError):
      await self.pipettes.request_y_position(channel=8)
    with self.assertRaises(ValueError):
      await self.pipettes.request_z_position(channel=-1)
    self.assertEqual(self.sent(), [])


class TestCoreGripperCommands(PositioningTestCase):
  """The CoRe gripper tool commands, written as the wire takes them."""

  async def test_discarding_a_tool(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.discard_core_gripper_tool(
      gripper_tool_x_position=100,
      first_gripper_tool_y_pos=200,
      tip_type=[14] * 8,
      begin_z_deposit_position=[0] * 8,
      end_z_deposit_position=[100] * 8,
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
      first_pip_channel_node_no=1,
      minimal_height_at_command_end=[2450] * 8,
    )
    await self.pipettes._unchecked_fw_discard_core_gripper_tool(
      x_position=100,
      first_tool_y_position=200,
      tip_types=[14] * 8,
      begin_z_deposit_positions=[0] * 8,
      end_z_deposit_positions=[100] * 8,
      minimal_traverse_heights=[2450] * 8,
      first_channel=1,
      minimal_heights_at_command_end=[2450] * 8,
    )
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_gripping_a_plate(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.grip_plate(
      plate_center_x_direction=100,
      plate_center_y_direction=200,
      plate_center_z_direction=300,
      z_speed=1287,
      open_gripper_position=860,
      plate_width=800,
      acceleration_index=4,
      grip_strength=30,
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
      minimal_height_at_command_end=[2450] * 8,
    )
    await self.pipettes._unchecked_fw_grip_core_gripper_plate(
      x_position=100,
      y_position=200,
      z_position=300,
      z_speed=1287,
      open_gripper_position=860,
      plate_width=800,
      acceleration_index=4,
      grip_strength=30,
      minimal_traverse_heights=[2450] * 8,
      minimal_heights_at_command_end=[2450] * 8,
    )
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_putting_a_plate(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.put_plate(
      plate_center_x_direction=100,
      plate_center_y_direction=200,
      plate_center_z_direction=300,
      press_on_distance=5,
      z_speed=1287,
      open_gripper_position=860,
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
      minimal_height_at_command_end=[2450] * 8,
    )
    await self.pipettes._unchecked_fw_put_core_gripper_plate(
      x_position=100,
      y_position=200,
      z_position=300,
      press_on_distance=5,
      z_speed=1287,
      open_gripper_position=860,
      minimal_traverse_heights=[2450] * 8,
      minimal_heights_at_command_end=[2450] * 8,
    )
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_moving_to_a_position(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.move_to_position(
      plate_center_x_direction=100,
      plate_center_y_direction=200,
      plate_center_z_direction=300,
      z_speed=1287,
      minimal_traverse_height_at_begin_of_command=[2450] * 8,
    )
    await self.pipettes._unchecked_fw_move_core_gripper_to_position(
      x_position=100,
      y_position=200,
      z_position=300,
      z_speed=1287,
      minimal_traverse_heights=[2450] * 8,
    )
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_releasing_an_object(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.release_object(first_pip_channel_node_no=3)
    await self.pipettes._unchecked_fw_release_core_gripper_object(first_channel=3)
    self.assertEqual(self.sent(), ["A1PMDOpa3"])
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))


class TestDispenseMode(unittest.TestCase):
  """Turning a jet, an empty and a blow-out into the firmware's dispense mode."""

  def test_a_jet_parts_the_liquid(self):
    self.assertEqual(_get_dispense_mode(jet=True, empty=False, blow_out=False), 0)

  def test_a_jet_with_a_blow_out(self):
    self.assertEqual(_get_dispense_mode(jet=True, empty=False, blow_out=True), 1)

  def test_liquid_stands_at_the_surface(self):
    self.assertEqual(_get_dispense_mode(jet=False, empty=False, blow_out=False), 2)

  def test_liquid_stands_at_the_surface_with_a_blow_out(self):
    self.assertEqual(_get_dispense_mode(jet=False, empty=False, blow_out=True), 3)

  def test_an_empty_empties_the_tip(self):
    self.assertEqual(_get_dispense_mode(jet=False, empty=True, blow_out=False), 4)

  def test_an_empty_ignores_a_jet_and_a_blow_out(self):
    self.assertEqual(_get_dispense_mode(jet=True, empty=True, blow_out=True), 4)


class TestRawAspirateDispense(PositioningTestCase):
  """The raw commands, sent exactly as they are given."""

  def eight(self, value):
    return [value] * 8

  async def test_the_raw_aspiration_command(self):
    await self.pipettes._unchecked_fw_aspirate(
      aspiration_type=self.eight(0),
      tip_pattern=[True] * 8,
      x_positions=self.eight(1000),
      y_positions=self.eight(3000),
      minimum_traverse_height_start=self.eight(2450),
      minimum_z_end_position=self.eight(2450),
      lld_search_heights=self.eight(0),
      clot_detection_heights=self.eight(60),
      liquid_surfaces_no_lld=self.eight(0),
      pull_out_distances_transport_air=self.eight(50),
      second_section_heights=self.eight(0),
      second_section_ratios=self.eight(0),
      minimum_heights=self.eight(3600),
      immersion_depths=self.eight(0),
      surface_following_distances=self.eight(0),
      aspiration_volumes=self.eight(100),
      aspiration_speeds=self.eight(500),
      transport_air_volumes=self.eight(0),
      blow_out_air_volumes=self.eight(0),
      pre_wetting_volumes=self.eight(0),
      lld_modes=self.eight(1),
      clld_sensitivities=self.eight(1),
      plld_sensitivities=self.eight(1),
      aspirate_positions_above_z_touch_off=self.eight(5),
      swap_speeds=self.eight(100),
      settling_times=self.eight(5),
      mix_volumes=self.eight(0),
      mix_cycles=self.eight(0),
      mix_positions_from_liquid_surface=self.eight(250),
      mix_speeds=self.eight(500),
      mix_surface_following_distances=self.eight(0),
      use_2nd_sections=[False] * 8,
      capacitive_mad_supervisions=[False] * 8,
      pressure_mad_supervisions=[False] * 8,
      tadm_algorithm=False,
      limit_curve_indices=self.eight(0),
      recording_mode=0,
    )
    self.assertEqual(
      self.sent(),
      [
        "A1PMDAat0 0 0 0 0 0 0 0tm1 1 1 1 1 1 1 1xp1000 1000 1000 1000 1000 1000 1000 1000"
        "yp3000 3000 3000 3000 3000 3000 3000 3000th2450 2450 2450 2450 2450 2450 2450 2450"
        "te2450 2450 2450 2450 2450 2450 2450 2450lp0 0 0 0 0 0 0 0"
        "ch60 60 60 60 60 60 60 60zl0 0 0 0 0 0 0 0po50 50 50 50 50 50 50 50"
        "zu0 0 0 0 0 0 0 0zr0 0 0 0 0 0 0 0zx3600 3600 3600 3600 3600 3600 3600 3600"
        "ip0 0 0 0 0 0 0 0fp0 0 0 0 0 0 0 0av100 100 100 100 100 100 100 100"
        "as500 500 500 500 500 500 500 500ta0 0 0 0 0 0 0 0ba0 0 0 0 0 0 0 0"
        "oa0 0 0 0 0 0 0 0lm1 1 1 1 1 1 1 1ll1 1 1 1 1 1 1 1lv1 1 1 1 1 1 1 1"
        "zo5 5 5 5 5 5 5 5de100 100 100 100 100 100 100 100wt5 5 5 5 5 5 5 5"
        "mv0 0 0 0 0 0 0 0mc0 0 0 0 0 0 0 0mp250 250 250 250 250 250 250 250"
        "ms500 500 500 500 500 500 500 500mh0 0 0 0 0 0 0 0la0 0 0 0 0 0 0 0"
        "lb0 0 0 0 0 0 0 0lc0 0 0 0 0 0 0 0gj0gi0 0 0 0 0 0 0 0gk0"
      ],
    )

  async def test_a_dispense_volume_is_written_with_four_digits(self):
    await self.pipettes._unchecked_fw_dispense(
      dispense_mode=self.eight(3),
      tip_pattern=[True] * 8,
      x_positions=self.eight(1000),
      y_positions=self.eight(3000),
      minimum_heights=self.eight(3600),
      lld_search_heights=self.eight(0),
      liquid_surfaces_no_lld=self.eight(0),
      pull_out_distances_transport_air=self.eight(50),
      immersion_depths=self.eight(0),
      surface_following_distances=self.eight(0),
      second_section_heights=self.eight(0),
      second_section_ratios=self.eight(0),
      minimum_traverse_height_start=self.eight(2450),
      minimum_z_end_position=self.eight(2450),
      dispense_volumes=self.eight(5),
      dispense_speeds=self.eight(1200),
      cut_off_speeds=self.eight(2500),
      stop_back_volumes=self.eight(0),
      transport_air_volumes=self.eight(50),
      blow_out_air_volumes=self.eight(0),
      lld_modes=self.eight(1),
      side_touch_off_distance=0,
      dispense_positions_above_z_touch_off=self.eight(5),
      clld_sensitivities=self.eight(1),
      plld_sensitivities=self.eight(1),
      swap_speeds=self.eight(100),
      settling_times=self.eight(5),
      mix_volumes=self.eight(0),
      mix_cycles=self.eight(0),
      mix_positions_from_liquid_surface=self.eight(250),
      mix_speeds=self.eight(500),
      mix_surface_following_distances=self.eight(0),
      use_2nd_sections=[False] * 8,
      tadm_algorithm=False,
      limit_curve_indices=self.eight(0),
      recording_mode=0,
    )
    self.assertEqual(
      self.sent(),
      [
        "A1PMDDdm3 3 3 3 3 3 3 3tm1 1 1 1 1 1 1 1xp1000 1000 1000 1000 1000 1000 1000 1000"
        "yp3000 3000 3000 3000 3000 3000 3000 3000"
        "zx3600 3600 3600 3600 3600 3600 3600 3600lp0 0 0 0 0 0 0 0"
        "zl0 0 0 0 0 0 0 0po50 50 50 50 50 50 50 50ip0 0 0 0 0 0 0 0"
        "fp0 0 0 0 0 0 0 0zu0 0 0 0 0 0 0 0zr0 0 0 0 0 0 0 0"
        "th2450 2450 2450 2450 2450 2450 2450 2450te2450 2450 2450 2450 2450 2450 2450 2450"
        "dv0005 0005 0005 0005 0005 0005 0005 0005"
        "ds1200 1200 1200 1200 1200 1200 1200 1200"
        "ss2500 2500 2500 2500 2500 2500 2500 2500rv0 0 0 0 0 0 0 0"
        "ta50 50 50 50 50 50 50 50ba0 0 0 0 0 0 0 0lm1 1 1 1 1 1 1 1dj0"
        "zo5 5 5 5 5 5 5 5ll1 1 1 1 1 1 1 1lv1 1 1 1 1 1 1 1"
        "de100 100 100 100 100 100 100 100wt5 5 5 5 5 5 5 5"
        "mv0 0 0 0 0 0 0 0mc0 0 0 0 0 0 0 0mp250 250 250 250 250 250 250 250"
        "ms500 500 500 500 500 500 500 500mh0 0 0 0 0 0 0 0la0 0 0 0 0 0 0 0"
        "gj0gi0 0 0 0 0 0 0 0gk0",
      ],
    )
