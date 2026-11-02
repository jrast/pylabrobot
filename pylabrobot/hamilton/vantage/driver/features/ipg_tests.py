import re
import unittest
from typing import Any, Dict, List

from pylabrobot.hamilton.vantage.driver.errors import VantageFirmwareError
from pylabrobot.hamilton.vantage.driver.features.ipg import Ipg, IpgParkConfiguration
from pylabrobot.hamilton.vantage.driver.features.pipettes_tests import LegacyCatcher
from pylabrobot.hamilton.vantage.driver.master import VantageDriver
from pylabrobot.hamilton.vantage.driver.master_tests import ScriptedIO, echo, machine


def without_id(command: str) -> str:
  """A command without the id it was sent with."""
  return re.sub(r"id\d+", "", command, count=1)


class IpgTestCase(unittest.IsolatedAsyncioTestCase):
  """A driver on a machine with an IPG, discovered."""

  async def asyncSetUp(self):
    self.io = ScriptedIO(machine(ipg=True))
    self.driver = VantageDriver(self.io, read_timeout=5)
    self.driver._connected = True
    self.driver._router.start()
    await self.driver.discover()
    self.io.written.clear()
    ipg = self.driver.ipg
    assert ipg is not None
    self.ipg: Ipg = ipg

  async def asyncTearDown(self):
    self.driver._router.stop()

  def sent(self):
    return [without_id(command) for command in self.io.written]


class TestQueries(IpgTestCase):
  """What the IPG is asked."""

  async def test_initialization_status(self):
    self.io.answers = machine(ipg=True, ipg_initialized=True)
    self.assertIs(await self.ipg.request_initialization_status(), True)
    self.io.answers = machine(ipg=True, ipg_initialized=False)
    self.assertIs(await self.ipg.request_initialization_status(), False)
    self.assertEqual(self.sent(), ["A1RMQW", "A1RMQW"])

  async def test_parking_status(self):
    self.io.answers = machine(ipg=True, ipg_parked=True)
    self.assertIs(await self.ipg.request_parking_status(), True)
    self.io.answers = machine(ipg=True, ipg_parked=False)
    self.assertIs(await self.ipg.request_parking_status(), False)
    self.assertEqual(self.sent(), ["A1RMRG", "A1RMRG"])

  async def test_plate_presence(self):
    self.io.answers = machine(ipg=True, ipg_plate=True)
    self.assertIs(await self.ipg.request_plate_presence(), True)
    self.io.answers = machine(ipg=True, ipg_plate=False)
    self.assertIs(await self.ipg.request_plate_presence(), False)
    self.assertEqual(self.sent(), ["A1RMQA", "A1RMQA"])

  async def test_angles(self):
    self.io.answers = lambda command: [echo(command, "rr45 271")]
    self.assertEqual(await self.ipg.request_angles(), [45, 271])
    self.assertEqual(self.sent(), ["A1RMRR"])

  async def test_park_configuration_is_in_mm_and_degrees(self):
    self.io.answers = lambda command: [echo(command, "rs6350 2879 726rr0 272")]
    self.assertEqual(
      await self.ipg.request_park_configuration(),
      IpgParkConfiguration(y=635.0, z=287.9, gripper_width=72.6, arm_angle=0, hand_angle=272),
    )
    self.assertEqual(self.sent(), ["A1RMRS"])


class TestBeforeTheArmIsUp(IpgTestCase):
  """The IPG cannot be initialized or parked until the arm has been initialized."""

  async def test_the_machine_refuses_and_says_why(self):
    self.io.answers = machine(ipg=True, arm_initialized=False)
    for move in (self.ipg.initialize, self.ipg.park):
      with self.assertRaises(VantageFirmwareError) as caught:
        await move()
      self.assertIn("MI", caught.exception.errors["IPG"])

  async def test_it_accepts_them_once_the_arm_is_up(self):
    self.io.answers = machine(ipg=True, arm_initialized=True)
    await self.ipg.initialize()
    await self.ipg.park()


class TestMoves(IpgTestCase):
  """Initializing and parking, which move the IPG."""

  async def test_initialize_sends_di_to_the_ipg(self):
    await self.ipg.initialize()
    self.assertEqual(self.sent(), ["A1RMDI"])

  async def test_park_sends_gp_to_the_ipg(self):
    await self.ipg.park()
    self.assertEqual(self.sent(), ["A1RMGP"])

  async def test_both_are_given_the_time_the_arm_waits_for_a_module(self):
    seen = {}
    send = self.driver._router.send

    async def spying(cmd, id_, write_timeout=None, read_timeout=None, wait=True):
      seen[cmd[:6]] = read_timeout
      return await send(cmd, id_, write_timeout, read_timeout, wait)

    self.driver._router.send = spying  # type: ignore[method-assign]
    await self.ipg.initialize()
    await self.ipg.park()
    self.assertEqual(seen, {"A1RMDI": 300, "A1RMGP": 300})


if __name__ == "__main__":
  unittest.main()


class TestGripOrientation(IpgTestCase):
  """Turning the wrist to the orientation the gripper grips at."""

  async def test_the_orientation_is_prepared_with_ga(self):
    await self.ipg.prepare_orientation()
    self.assertEqual(self.sent(), ["A1RMGAgd32th2450"])

  async def test_it_matches_the_legacy_command(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.ipg_prepare_gripper_orientation(
      grip_orientation=11, minimal_traverse_height_at_begin_of_command=3000
    )
    await self.ipg.prepare_orientation(grip_orientation=11, minimal_traverse_height=300.0)
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_both_digits_of_the_orientation_are_checked(self):
    for orientation in (0, 9, 10, 19, 89, 100):
      with self.assertRaises(ValueError):
        await self.ipg.prepare_orientation(grip_orientation=orientation)
    self.assertEqual(self.sent(), [])

  async def test_the_height_is_checked(self):
    with self.assertRaises(ValueError):
      await self.ipg.prepare_orientation(minimal_traverse_height=400.1)
    self.assertEqual(self.sent(), [])


class TestMoveTo(IpgTestCase):
  """Travelling to a place."""

  async def test_a_move_goes_to_the_place_at_the_traverse_height(self):
    await self.ipg.move_to(127.5, 85.0, 100.0)
    self.assertEqual(self.sent(), ["A1RMDNxp1275yp850zp1000th2450"])

  async def test_it_matches_the_legacy_command(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.ipg_move_to_defined_position(
      x_position=-1234,
      y_position=567,
      z_position=890,
      minimal_traverse_height_at_begin_of_command=3000,
    )
    await self.ipg.move_to(-123.4, 56.7, 89.0, minimal_traverse_height=300.0)
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_a_place_out_of_reach_is_refused(self):
    for place in ((6000.0, 0.0, 0.0), (0.0, -1000.1, 0.0), (0.0, 0.0, -0.1), (0.0, 0.0, 400.1)):
      with self.assertRaises(ValueError):
        await self.ipg.move_to(*place)
    self.assertEqual(self.sent(), [])


class TestGripPlate(IpgTestCase):
  """Gripping a plate, which is a `DN` and then a `DG`."""

  async def test_a_grip_is_refused_before_the_orientation_is_prepared(self):
    with self.assertRaises(ValueError) as caught:
      await self.ipg.grip_plate(1.0, 2.0, 3.0, plate_width=127.8)
    self.assertIn("prepare_orientation", str(caught.exception))
    self.assertEqual(self.sent(), [])

  async def test_a_grip_travels_there_then_grips(self):
    await self.ipg.prepare_orientation()
    self.io.written.clear()
    await self.ipg.grip_plate(127.5, 85.0, 100.0, plate_width=127.8)
    self.assertEqual(
      self.sent(),
      [
        "A1RMDNxp1275yp850zp1000th2450",
        "A1RMDGxp1275yp850zp1000yw81yo1310yg1245pt20ai4zc0hd0te2450",
      ],
    )

  async def test_it_matches_the_legacy_command(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.ipg_grip_plate(
      x_position=1275,
      y_position=850,
      z_position=1000,
      grip_strength=100,
      open_gripper_position=1300,
      plate_width=1200,
      plate_width_tolerance=10,
      acceleration_index=2,
      z_clearance_height=5,
      hotel_depth=10,
      minimal_height_at_command_end=3000,
    )
    await self.ipg.prepare_orientation()
    await self.ipg.grip_plate(
      127.5,
      85.0,
      100.0,
      plate_width=120.0,
      open_gripper_width=130.0,
      grip_width=120.0,
      grip_strength=100,
      plate_width_tolerance=1.0,
      acceleration_index=2,
      z_clearance_height=0.5,
      hotel_depth=1.0,
      minimal_height_at_command_end=300.0,
    )
    self.assertEqual(self.sent()[-1], without_id(catcher.commands[-1]))

  async def test_the_gripper_opens_and_closes_around_the_plate(self):
    await self.ipg.prepare_orientation()
    self.io.written.clear()
    await self.ipg.grip_plate(10.0, 20.0, 30.0, plate_width=85.5)
    self.assertEqual(self.sent()[-1], "A1RMDGxp100yp200zp300yw81yo887yg822pt20ai4zc0hd0te2450")

  async def test_values_out_of_range_are_refused(self):
    await self.ipg.prepare_orientation()
    self.io.written.clear()
    refused: List[Dict[str, Any]] = [
      {"plate_width": 1000.0},
      {"plate_width": 100.0, "grip_strength": 161},
      {"plate_width": 100.0, "plate_width_tolerance": 10.0},
      {"plate_width": 100.0, "acceleration_index": 5},
      {"plate_width": 100.0, "z_clearance_height": 100.0},
      {"plate_width": 100.0, "hotel_depth": 301.0},
      {"plate_width": 100.0, "minimal_height_at_command_end": 400.1},
    ]
    for value in refused:
      with self.assertRaises(ValueError):
        await self.ipg.grip_plate(10.0, 20.0, 30.0, **value)
    self.assertEqual(self.sent(), [])

  async def test_parking_forgets_the_orientation(self):
    await self.ipg.prepare_orientation()
    await self.ipg.park()
    with self.assertRaises(ValueError):
      await self.ipg.grip_plate(10.0, 20.0, 30.0, plate_width=100.0)

  async def test_initializing_forgets_the_orientation(self):
    await self.ipg.prepare_orientation()
    await self.ipg.initialize()
    with self.assertRaises(ValueError):
      await self.ipg.grip_plate(10.0, 20.0, 30.0, plate_width=100.0)


class TestPutPlate(IpgTestCase):
  """Putting the plate down, which is a `DN` and then a `DR`."""

  async def test_a_put_is_refused_before_the_orientation_is_prepared(self):
    with self.assertRaises(ValueError) as caught:
      await self.ipg.put_plate(1.0, 2.0, 3.0, plate_width=127.8)
    self.assertIn("prepare_orientation", str(caught.exception))
    self.assertEqual(self.sent(), [])

  async def test_a_put_travels_there_then_puts(self):
    await self.ipg.prepare_orientation()
    self.io.written.clear()
    await self.ipg.put_plate(127.5, 85.0, 100.0, plate_width=127.8)
    self.assertEqual(
      self.sent(),
      [
        "A1RMDNxp1275yp850zp1000th2450",
        "A1RMDRxp1275yp850zp1000yo1310zc0zi5hd0te2450",
      ],
    )

  async def test_it_is_the_legacy_command_with_the_press_on_distance(self):
    catcher = LegacyCatcher(num_channels=8)
    await catcher.ipg_put_plate(
      x_position=1275,
      y_position=850,
      z_position=1000,
      open_gripper_position=1300,
      z_clearance_height=5,
      press_on_distance=5,
      hotel_depth=10,
      minimal_height_at_command_end=3000,
    )
    await self.ipg.prepare_orientation()
    await self.ipg.put_plate(
      127.5,
      85.0,
      100.0,
      plate_width=120.0,
      open_gripper_width=130.0,
      z_clearance_height=0.5,
      hotel_depth=1.0,
      minimal_height_at_command_end=300.0,
    )
    legacy = without_id(catcher.commands[-1])
    self.assertEqual(self.sent()[-1], legacy.replace("zc5hd", "zc5zi5hd"))

  async def test_the_press_on_distance_is_taken_in_mm(self):
    await self.ipg.prepare_orientation()
    self.io.written.clear()
    await self.ipg.put_plate(10.0, 20.0, 30.0, plate_width=100.0, press_on_distance=1.2)
    self.assertEqual(self.sent()[-1], "A1RMDRxp100yp200zp300yo1032zc0zi12hd0te2450")

  async def test_values_out_of_range_are_refused(self):
    await self.ipg.prepare_orientation()
    self.io.written.clear()
    refused: List[Dict[str, Any]] = [
      {"plate_width": 1000.0},
      {"plate_width": 100.0, "press_on_distance": 100.0},
      {"plate_width": 100.0, "hotel_depth": 301.0},
    ]
    for value in refused:
      with self.assertRaises(ValueError):
        await self.ipg.put_plate(10.0, 20.0, 30.0, **value)
    self.assertEqual(self.sent(), [])


class TestReachQueries(IpgTestCase):
  """The two questions about where the IPG can reach, whose answers are not read yet."""

  async def test_access_range(self):
    self.io.answers = lambda command: [echo(command, 'ra"reach"')]
    reply = await self.ipg.request_access_range()
    self.assertEqual(self.sent(), ["A1RMQRgd32"])
    self.assertIn('er0ra"reach"', reply)

  async def test_position(self):
    self.io.answers = lambda command: [echo(command, 'ri"place"')]
    reply = await self.ipg.request_position(grip_orientation=99)
    self.assertEqual(self.sent(), ["A1RMQIgd99"])
    self.assertIn('er0ri"place"', reply)

  async def test_the_access_range_takes_both_digits_up_to_eight(self):
    for orientation in (0, 19, 89, 100):
      with self.assertRaises(ValueError):
        await self.ipg.request_access_range(grip_orientation=orientation)
    self.assertEqual(self.sent(), [])

  async def test_the_position_takes_both_digits_up_to_nine(self):
    self.io.answers = lambda command: [echo(command)]
    await self.ipg.request_position(grip_orientation=19)
    await self.ipg.request_position(grip_orientation=99)
    self.assertEqual(self.sent(), ["A1RMQIgd19", "A1RMQIgd99"])
    for orientation in (0, 100):
      with self.assertRaises(ValueError):
        await self.ipg.request_position(grip_orientation=orientation)
