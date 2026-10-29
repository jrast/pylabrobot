"""The Vantage's pipetting channels, which the arm addresses all together as `A1PM`."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, cast

from pylabrobot.resources.coordinate import Coordinate
from pylabrobot.resources.errors import HasTipError, NoTipError
from pylabrobot.resources.hamilton.tip_creators import HamiltonTip
from pylabrobot.resources.tip_rack import TipSpot

if TYPE_CHECKING:
  from pylabrobot.hamilton.vantage.driver.master import VantageDriver

logger = logging.getLogger(__name__)


@dataclass
class PipettesConfiguration:
  """What the channels accept, and how a value in mm is written on the wire."""

  increments_per_mm: int = 10
  """The wire carries positions and heights in tenths of a millimeter."""

  x_range: Tuple[float, float] = (0.0, 5000.0)
  """Along X, in mm."""

  y_range: Tuple[float, float] = (0.0, 650.0)
  """Along Y, in mm."""

  z_range: Tuple[float, float] = (0.0, 360.0)
  """Along Z, in mm."""

  x_shift_range: Tuple[float, float] = (-100.0, 100.0)
  """How far a channel may shift along X while it discards a tip, in mm."""

  tip_type_range: Tuple[int, int] = (0, 199)
  """The indexes of the machine's tip type table."""

  default_minimal_traverse_height: float = 245.0
  """How high a tip command travels with the channels when nothing else is asked, in mm."""

  minimal_traverse_height: float = 245.0
  """The lowest a tip command may travel at, in mm. Up to this height a resource is under a
  traversing channel whatever tip it carries, so travelling lower may drag a tip through the deck."""

  minimum_y_spacing: float = 9.0
  """How close together two channels may stand, in mm. A 96-well rack's columns are exactly this,
  so tips from one column can be taken together; anything tighter the head cannot reach."""

  initialize_read_timeout: int = 300
  """How long to wait for the channels to initialize, in seconds. They go to the waste, discard
  what they carry and initialize their drives, and the arm itself waits that long for them."""

  def to_increments(self, mm: float) -> int:
    """A value in mm as the wire carries it."""
    return round(mm * self.increments_per_mm)


def _check_traverse_height(height: float, config: "PipettesConfiguration") -> None:
  """Raise unless a tip command may travel at this height.

  Raises:
    ValueError: If the height is below what clears the deck, or above what the channels reach.
  """
  if height < config.minimal_traverse_height:
    raise ValueError(
      f"a tip command travels no lower than {config.minimal_traverse_height} mm, not {height}"
    )
  _check_range("minimum_traverse_height_start", [height], config.z_range)


def _check_range(name: str, values: List[float], limits: Tuple[float, float]) -> None:
  """Raise unless every value is within the limits, both ends included."""
  low, high = limits
  for value in values:
    if not low <= value <= high:
      raise ValueError(f"{name} must be from {low} to {high}, but is {value}")


def _get_dispense_mode(jet: bool, empty: bool, blow_out: bool) -> int:
  """The dispense mode (`dm`) for a jet, an empty and a blow-out.

  The firmware's names and the VENUS liquid editor's differ: what the editor calls "empty" is a
  blow-out, while the firmware's "empty" (4) empties the tip completely. A blow-out at the surface
  is the mode used when neither a jet nor an empty is asked for.

  Args:
    jet: whether the liquid leaves as a jet, as opposed to standing at the surface.
    empty: whether to empty the tip completely.
    blow_out: whether to blow out after the liquid.

  Returns:
    The firmware's mode: 0 part in jet, 1 blow in jet, 2 part at surface, 3 blow at surface,
    4 empty.
  """
  if empty:
    return 4
  if jet:
    return 1 if blow_out else 0
  return 3 if blow_out else 2


class Pipettes:
  """The pipetting channels.

  Reached as `driver.pipettes`. The channels are addressed together, so every list a command
  takes holds one value for each channel the machine has.

  Every place these channels move to comes from the deck model, and the driver applies no
  correction of its own: a place that is wrong is a deck-model bug, fixed where the model is built.
  """

  default_x_position: float = 709.5
  """Where the channels go to discard what they carry when they initialize, along X, in mm."""

  default_y_positions: Dict[int, List[float]] = {
    8: [389.1, 362.3, 335.5, 308.7, 281.9, 255.1, 228.3, 201.6],
  }
  """Where each channel goes to discard, along Y, in mm, by how many channels the machine has. A
  machine whose count is not here must be told."""

  default_begin_of_tip_deposit: float = 245.0
  """Where the channels begin to deposit a tip, along Z, in mm."""

  default_end_of_tip_deposit: float = 123.5
  """Where the channels stop depositing it, along Z, in mm."""

  default_minimal_height_at_command_end: float = 245.0
  """How high the channels end initialization, along Z, in mm."""

  default_tip_type: int = 4
  """The tip type the channels initialize for. The machine wants the longest tip in use, and index 4
  is the 1000 ul tip without a filter."""

  default_x_shift_during_discard: float = 7.0
  """How far the channels shift along X while discarding, to make a tip fall into the waste, in
  mm."""

  def __init__(self, driver: "VantageDriver"):
    self._driver = driver
    self.configuration = PipettesConfiguration()

    # The tip each channel carries, by channel. What the model says, not what the machine senses:
    # `sense_tip_presence` asks the machine.
    self._mounted_tips: Dict[int, HamiltonTip] = {}

  @property
  def num_channels(self) -> int:
    """How many channels the machine has. Read from it by `VantageDriver.discover`."""
    return self._driver.num_channels

  async def sense_tip_presence(self) -> List[int]:
    """Sense tip presence on every channel, from the machine's own sensors.

    Answered as the machine answers it, 1 where a tip is and 0 where none is, rather than narrowed
    to True and False: the two carry the same meaning, and a value that is neither would be lost by
    the narrowing rather than read back as it stands.

    Returns:
      One value per channel, 1 where a tip is mounted, 0-indexed from the back.
    """
    resp = await self._driver.send_command(module="A1PM", command="QA", fmt={"rt": "[int]"})
    return cast(List[int], resp["rt"])

  async def request_initialization_status(self) -> bool:
    """Whether the channels are initialized."""
    resp = await self._driver.send_command(module="A1PM", command="QW", fmt={"qw": "int"})
    return bool(resp["qw"] == 1)

  async def _unchecked_fw_initialize(
    self,
    x_position: List[int],
    y_position: List[int],
    begin_z_deposit_position: List[int],
    end_z_deposit_position: List[int],
    minimal_height_at_command_end: List[int],
    tip_pattern: List[bool],
    tip_type: List[int],
    x_shift: int,
  ) -> None:
    """Send the initialization as it goes on the wire, with every value in tenths of a mm and
    nothing checked."""
    await self._driver.send_command(
      module="A1PM",
      command="DI",
      read_timeout=self.configuration.initialize_read_timeout,
      xp=x_position,
      yp=y_position,
      tp=begin_z_deposit_position,
      tz=end_z_deposit_position,
      te=minimal_height_at_command_end,
      tm=tip_pattern,
      tt=tip_type,
      ts=x_shift,
    )

  def get_mounted_tip(self, channel: int) -> Optional[HamiltonTip]:
    """The tip the model has on a channel, or None if it carries none.

    Args:
      channel: which channel, 0-indexed from the back.
    """
    return self._mounted_tips.get(channel)

  def _channels_and_offsets(
    self,
    spots: Sequence[TipSpot],
    use_channels: Optional[List[int]],
    offsets: Optional[List[Coordinate]],
  ) -> Tuple[List[int], List[Coordinate]]:
    """Which channels take which spot, and how far from each spot's centre they go, checked.

    Args:
      spots: one spot per channel, in ascending channel order.
      use_channels: which channels, 0-indexed from the back. The first `len(spots)` when None.
      offsets: added to each spot's centre, in mm.

    Returns:
      The channels and their offsets.

    Raises:
      ValueError: If the channels are not ascending, a channel is repeated or not fitted, or the
        spots and channels do not come in pairs.
    """
    channels = list(range(len(spots))) if use_channels is None else list(use_channels)
    if len(channels) != len(spots):
      raise ValueError(
        f"one channel for each spot is needed, but {len(spots)} spots and "
        f"{len(channels)} channels are given"
      )
    if channels != sorted(channels):
      raise ValueError(f"the channels must be ascending, are {channels}")
    if len(set(channels)) != len(channels):
      raise ValueError(f"a channel can take only one tip, but {channels} lists one twice")
    if any(channel < 0 or channel >= self.num_channels for channel in channels):
      raise ValueError(f"the machine has channels 0 to {self.num_channels - 1}, not {channels}")
    return channels, [Coordinate.zero()] * len(spots) if offsets is None else list(offsets)

  def _spot_positions(
    self,
    spots: Sequence[TipSpot],
    channels: List[int],
    offsets: List[Coordinate],
  ) -> Tuple[List[int], List[int]]:
    """Where each channel goes, in tenths of a mm, checked against what the channels can reach.

    Args:
      spots: one spot per channel, in the order of `channels`.
      channels: which channels, ascending.
      offsets: added to each spot's centre, in mm.

    Returns:
      X and Y, one for each channel given.

    Raises:
      RuntimeError: If the driver has no deck.
      ValueError: If a position is out of reach.
    """
    deck = self._driver.deck
    if deck is None:
      raise RuntimeError("tips are picked up at places on the deck; this driver was given none")
    config = self.configuration
    xs: List[int] = []
    ys: List[int] = []
    for spot, offset in zip(spots, offsets):
      location = spot.get_location_wrt(deck, x="c", y="c", z="b") + offset
      _check_range("x", [location.x], config.x_range)
      _check_range("y", [location.y], config.y_range)
      xs.append(config.to_increments(location.x))
      ys.append(config.to_increments(location.y))

    # Channels stand no closer than the machine lets them. Two places in one column closer than
    # that cannot be taken together, and the machine works through them one at a time instead, so
    # the run takes longer than asked. Worth saying, but the tips still come up.
    least = config.to_increments(config.minimum_y_spacing)
    for first in range(len(xs)):
      for second in range(first + 1, len(xs)):
        if xs[first] == xs[second] and abs(ys[first] - ys[second]) < least:
          logger.warning(
            "two places in one column are %.1f mm apart, closer than the %.1f mm this machine lets "
            "the channels stand: it takes them one at a time, so this takes longer",
            abs(ys[first] - ys[second]) / config.increments_per_mm,
            config.minimum_y_spacing,
          )
    return xs, ys

  def _tips_to_collect(self, spots: Sequence[TipSpot]) -> List[HamiltonTip]:
    """What each spot gives up, checked to be one model of Hamilton tip.

    Raises:
      TypeError: If a spot holds something other than a Hamilton tip.
      ValueError: If the tips are not all of one model.
      NoTipError: If a spot holds no tip while tip tracking is on.
    """
    tips: List[HamiltonTip] = []
    for spot in spots:
      tip = spot.tip_for_pickup()
      if not isinstance(tip, HamiltonTip):
        raise TypeError("the Vantage picks up Hamilton tips")
      tips.append(tip)
    if len({tip.model for tip in tips}) > 1:
      raise ValueError("the tips picked up in one command must all have the same model")
    return tips

  def _tip_pattern(self, channels: List[int]) -> List[bool]:
    """Which channels of the machine take part, one entry for each of them."""
    return [channel in channels for channel in range(self.num_channels)]

  def _full_width(self, channels: List[int], values: List) -> List:
    """One value for each channel of the machine, with 0 for the channels not given.

    A list shorter than the machine's channels would not leave the rest at 0: the machine takes a
    `&` after it as "the last value again for every channel that is left", so the channels that
    take no part would be given the last one's value. Every value is written out instead.
    """
    full: List = [0] * self.num_channels
    for channel, value in zip(channels, values):
      full[channel] = value
    return full

  def _per_channel(self, name: str, values: Optional[List], default) -> List:
    """What was given for each channel, or the default for every channel.

    Raises:
      ValueError: If a list does not hold one value for each channel.
    """
    if values is None:
      return [default] * self.num_channels
    if len(values) != self.num_channels:
      raise ValueError(
        f"{name} must hold one value for each of the {self.num_channels} channels, "
        f"but holds {len(values)}"
      )
    return list(values)

  def _default_y_positions(self) -> List[float]:
    """Where each channel discards by default.

    Raises:
      ValueError: If the machine's channel count has no default.
    """
    defaults = self.default_y_positions.get(self.num_channels)
    if defaults is None:
      known = ", ".join(str(count) for count in sorted(self.default_y_positions))
      raise ValueError(
        f"there is no default discard position along Y for {self.num_channels} channels (only "
        f"for {known}): pass `y_positions`, one for each channel, in mm"
      )
    return list(defaults)

  async def initialize(
    self,
    x_position: Optional[List[float]] = None,
    y_positions: Optional[List[float]] = None,
    begin_of_tip_deposit: Optional[List[float]] = None,
    end_of_tip_deposit: Optional[List[float]] = None,
    minimal_height_at_command_end: Optional[List[float]] = None,
    tip_pattern: Optional[List[bool]] = None,
    tip_type: Optional[List[int]] = None,
    x_shift_during_discard: Optional[float] = None,
  ) -> None:
    """Initialize the channels. They go to the waste, discard what they carry, and initialize
    their drives. Moves the channels.

    Every argument but the shift holds one value for each channel, in mm. What is not given is
    taken from the `default_*` attributes.

    Args:
      x_position: where each channel discards, along X.
      y_positions: where each channel discards, along Y. Without it the machine's channel count
        needs a default.
      begin_of_tip_deposit: where each channel begins to deposit its tip, along Z.
      end_of_tip_deposit: where each channel stops depositing it, along Z.
      minimal_height_at_command_end: how high each channel ends, along Z.
      tip_pattern: which channels are initialized. All of them by default.
      tip_type: the tip type of each channel, an index of the machine's tip type table.
      x_shift_during_discard: how far the channels shift along X while they discard, in mm.

    Raises:
      ValueError: If a list does not hold one value for each channel, a value is out of range, or
        `y_positions` is needed and not given. Nothing is sent then.
    """
    config = self.configuration
    x = self._per_channel("x_position", x_position, self.default_x_position)
    y = (
      self._per_channel("y_positions", y_positions, 0.0)
      if y_positions is not None
      else self._default_y_positions()
    )
    begin = self._per_channel(
      "begin_of_tip_deposit", begin_of_tip_deposit, self.default_begin_of_tip_deposit
    )
    end = self._per_channel(
      "end_of_tip_deposit", end_of_tip_deposit, self.default_end_of_tip_deposit
    )
    height = self._per_channel(
      "minimal_height_at_command_end",
      minimal_height_at_command_end,
      self.default_minimal_height_at_command_end,
    )
    pattern = self._per_channel("tip_pattern", tip_pattern, True)
    types = self._per_channel("tip_type", tip_type, self.default_tip_type)
    shift = (
      self.default_x_shift_during_discard
      if x_shift_during_discard is None
      else x_shift_during_discard
    )

    _check_range("x_position", x, config.x_range)
    _check_range("y_positions", y, config.y_range)
    _check_range("begin_of_tip_deposit", begin, config.z_range)
    _check_range("end_of_tip_deposit", end, config.z_range)
    _check_range("minimal_height_at_command_end", height, config.z_range)
    _check_range("tip_type", types, config.tip_type_range)  # type: ignore[arg-type]
    _check_range("x_shift_during_discard", [shift], config.x_shift_range)

    await self._unchecked_fw_initialize(
      x_position=[config.to_increments(value) for value in x],
      y_position=[config.to_increments(value) for value in y],
      begin_z_deposit_position=[config.to_increments(value) for value in begin],
      end_z_deposit_position=[config.to_increments(value) for value in end],
      minimal_height_at_command_end=[config.to_increments(value) for value in height],
      tip_pattern=[bool(value) for value in pattern],
      tip_type=[int(value) for value in types],
      x_shift=config.to_increments(shift),
    )

  # -- positioning ---------------------------------------------------------

  def _check_channel(self, channel: int) -> None:
    """Raise unless the channel is one the machine has.

    Raises:
      ValueError: If it is not.
    """
    if not 0 <= channel < self.num_channels:
      raise ValueError(f"channel must be from 0 to {self.num_channels - 1}, but is {channel}")

  def _require_all_channels(self, positions: Dict[int, float], name: str) -> None:
    """Raise unless every channel has a value, and no key names anything else.

    Raises:
      ValueError: If a channel is missing, or a key is not one.
    """
    missing = [channel for channel in range(self.num_channels) if channel not in positions]
    if missing:
      raise ValueError(
        f"`{name}` must give a value for every one of the {self.num_channels} channels, since the "
        f"machine moves them all in one command; missing {missing}"
      )
    unknown = [channel for channel in positions if not 0 <= channel < self.num_channels]
    if unknown:
      raise ValueError(f"`{name}` names channels this machine does not have: {unknown}")

  def _all_positions_in_increments(
    self, positions: Dict[int, float], name: str, limits: Tuple[float, float]
  ) -> List[int]:
    """Every channel's position in wire increments, checked and full width.

    Raises:
      ValueError: If a channel is missing, a key is not a channel, or a position is out of range.
    """
    self._require_all_channels(positions, name)
    values = [positions[channel] for channel in range(self.num_channels)]
    _check_range(name, values, limits)
    return [self.configuration.to_increments(value) for value in values]

  async def move_to_y_positions(self, y_positions: Dict[int, float]) -> None:
    """Move every channel along Y. Moves the channels.

    The machine takes one position for every channel in one command, and there is no pattern to
    leave one out, so every channel has to be given: a channel left out would be sent to a
    position nobody asked for. `move_to_y_position` moves one on its own.

    Args:
      y_positions: where to put each channel, in mm, keyed by channel, 0-indexed from the back.

    Raises:
      ValueError: If a channel is missing, a key is not a channel, or a position is out of range.
    """
    await self._driver.send_command(
      module="A1PM",
      command="DY",
      yp=self._all_positions_in_increments(y_positions, "y_positions", self.configuration.y_range),
    )

  async def move_to_z_positions(self, z_positions: Dict[int, float]) -> None:
    """Move every channel along Z. Moves the channels.

    As `move_to_y_positions`, every channel has to be given.

    Args:
      z_positions: where to put each channel, in mm, keyed by channel, 0-indexed from the back.

    Raises:
      ValueError: If a channel is missing, a key is not a channel, or a position is out of range.
    """
    await self._driver.send_command(
      module="A1PM",
      command="DZ",
      zp=self._all_positions_in_increments(z_positions, "z_positions", self.configuration.z_range),
    )

  async def move_to_y_position(self, channel: int, y: float) -> None:
    """Move one channel along Y. Moves the channel.

    Args:
      channel: which channel, 0-indexed from the back.
      y: where to put it, in mm.

    Raises:
      ValueError: If the channel is not one the machine has, or the position is out of range.
    """
    self._check_channel(channel)
    _check_range("y", [y], self.configuration.y_range)
    await self._driver.send_command(
      module="A1PM", command="DV", pn=channel + 1, yj=self.configuration.to_increments(y)
    )

  async def move_to_z_position(self, channel: int, z: float) -> None:
    """Move one channel along Z. Moves the channel.

    Args:
      channel: which channel, 0-indexed from the back.
      z: where to put it, in mm.

    Raises:
      ValueError: If the channel is not one the machine has, or the position is out of range.
    """
    self._check_channel(channel)
    _check_range("z", [z], self.configuration.z_range)
    await self._driver.send_command(
      module="A1PM", command="DU", pn=channel + 1, zj=self.configuration.to_increments(z)
    )

  async def move_to_positions(
    self,
    xs: Dict[int, float],
    ys: Dict[int, float],
    zs: Dict[int, float],
    *,
    minimal_traverse_height: Optional[float] = None,
  ) -> None:
    """Move the given channels to places of their own. Moves the channels.

    All three axes go in one command, with a pattern that names the channels it is about: the
    channels it does not name keep their positions and travel nowhere.

    Args:
      xs: where to put each named channel, along X, in mm.
      ys: where to put each named channel, along Y, in mm.
      zs: where to put each named channel, along Z, in mm.
      minimal_traverse_height: how high the channels travel first, in mm. The configured default
        when None, and never below what clears the deck.

    Raises:
      ValueError: If the three do not name the same channels, nothing is named, a channel is not
        one the machine has, a position is out of range, or the traverse height is.
    """
    channels = set(xs) | set(ys) | set(zs)
    if not channels:
      raise ValueError("nothing to move: give the places of at least one channel")
    for name, positions in (("xs", xs), ("ys", ys), ("zs", zs)):
      if set(positions) != channels:
        raise ValueError(
          f"`{name}` must name the same channels as the other axes, "
          f"{sorted(channels)}, but names {sorted(positions)}"
        )
    ordered = sorted(channels)
    for channel in ordered:
      self._check_channel(channel)

    config = self.configuration
    _check_range("xs", [xs[channel] for channel in ordered], config.x_range)
    _check_range("ys", [ys[channel] for channel in ordered], config.y_range)
    _check_range("zs", [zs[channel] for channel in ordered], config.z_range)
    travel = (
      config.default_minimal_traverse_height
      if minimal_traverse_height is None
      else minimal_traverse_height
    )
    _check_traverse_height(travel, config)

    await self._driver.send_command(
      module="A1PM",
      command="DN",
      tm=self._tip_pattern(ordered),
      xp=self._full_width(ordered, [config.to_increments(xs[channel]) for channel in ordered]),
      yp=self._full_width(ordered, [config.to_increments(ys[channel]) for channel in ordered]),
      # The height is given for every channel, as the machine takes it, though only the channels
      # in the pattern move.
      th=[config.to_increments(travel)] * self.num_channels,
      zp=self._full_width(ordered, [config.to_increments(zs[channel]) for channel in ordered]),
    )

  async def expose_channel(self, channel: int) -> None:
    """Move the channels aside so that one stands exposed, within reach by hand. Moves them.

    Args:
      channel: which channel to expose, 0-indexed from the back.

    Raises:
      ValueError: If the channel is not one the machine has.
    """
    self._check_channel(channel)
    await self._driver.send_command(module="A1PM", command="DQ", pn=channel + 1)

  async def request_y_positions(self) -> str:
    """Ask where every channel stands along Y.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.
    """
    return await self._driver.send_command(module="A1PM", command="RY")

  async def request_y_position(self, channel: int) -> str:
    """Ask where one channel stands along Y.

    Args:
      channel: which channel, 0-indexed from the back.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.

    Raises:
      ValueError: If the channel is not one the machine has.
    """
    self._check_channel(channel)
    return await self._driver.send_command(module="A1PM", command="RB", pn=channel + 1)

  async def request_z_positions(self) -> str:
    """Ask where every channel stands along Z.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.
    """
    return await self._driver.send_command(module="A1PM", command="RZ")

  async def request_z_position(self, channel: int) -> str:
    """Ask where one channel stands along Z.

    Args:
      channel: which channel, 0-indexed from the back.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.

    Raises:
      ValueError: If the channel is not one the machine has.
    """
    self._check_channel(channel)
    return await self._driver.send_command(module="A1PM", command="RD", pn=channel + 1)

  async def request_height_of_last_lld(self) -> str:
    """Ask how high the liquid level of the last measurement was.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.
    """
    return await self._driver.send_command(module="A1PM", command="RL")

  async def request_dispense_on_fly_status(self) -> str:
    """Ask what the channels report about dispensing on the fly.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.
    """
    return await self._driver.send_command(module="A1PM", command="QF")

  # -- CoRe gripper tools, as the wire takes them ---------------------------

  async def _unchecked_fw_discard_core_gripper_tool(
    self,
    x_position: int,
    first_tool_y_position: int,
    tip_types: List[int],
    begin_z_deposit_positions: List[int],
    end_z_deposit_positions: List[int],
    minimal_traverse_heights: List[int],
    first_channel: int,
    minimal_heights_at_command_end: List[int],
  ) -> None:
    """Discard the CoRe gripper tool each channel carries (`DJ`).

    The positions are in tenths of a millimeter, as the wire takes them, and one value for each
    channel; nothing is checked here.
    """
    await self._driver.send_command(
      module="A1PM",
      command="DJ",
      xa=x_position,
      yj=first_tool_y_position,
      tt=tip_types,
      tp=begin_z_deposit_positions,
      tz=end_z_deposit_positions,
      th=minimal_traverse_heights,
      pa=first_channel,
      te=minimal_heights_at_command_end,
    )

  async def _unchecked_fw_grip_core_gripper_plate(
    self,
    x_position: int,
    y_position: int,
    z_position: int,
    z_speed: int,
    open_gripper_position: int,
    plate_width: int,
    acceleration_index: int,
    grip_strength: int,
    minimal_traverse_heights: List[int],
    minimal_heights_at_command_end: List[int],
  ) -> None:
    """Grip a plate onto the CoRe gripper tools (`DG`), in tenths of a millimeter."""
    await self._driver.send_command(
      module="A1PM",
      command="DG",
      xa=x_position,
      yj=y_position,
      zj=z_position,
      zy=z_speed,
      yo=open_gripper_position,
      yg=plate_width,
      ai=acceleration_index,
      yw=grip_strength,
      th=minimal_traverse_heights,
      te=minimal_heights_at_command_end,
    )

  async def _unchecked_fw_put_core_gripper_plate(
    self,
    x_position: int,
    y_position: int,
    z_position: int,
    press_on_distance: int,
    z_speed: int,
    open_gripper_position: int,
    minimal_traverse_heights: List[int],
    minimal_heights_at_command_end: List[int],
  ) -> None:
    """Put the plate the CoRe gripper tools carry down (`DR`), in tenths of a millimeter."""
    await self._driver.send_command(
      module="A1PM",
      command="DR",
      xa=x_position,
      yj=y_position,
      zj=z_position,
      zi=press_on_distance,
      zy=z_speed,
      yo=open_gripper_position,
      th=minimal_traverse_heights,
      te=minimal_heights_at_command_end,
    )

  async def _unchecked_fw_move_core_gripper_to_position(
    self,
    x_position: int,
    y_position: int,
    z_position: int,
    z_speed: int,
    minimal_traverse_heights: List[int],
  ) -> None:
    """Move the CoRe gripper tools to a place (`DH`), in tenths of a millimeter."""
    await self._driver.send_command(
      module="A1PM",
      command="DH",
      xa=x_position,
      yj=y_position,
      zj=z_position,
      zy=z_speed,
      th=minimal_traverse_heights,
    )

  async def _unchecked_fw_release_core_gripper_object(self, first_channel: int) -> None:
    """Let go of what the CoRe gripper tools hold (`DO`)."""
    await self._driver.send_command(module="A1PM", command="DO", pa=first_channel)

  # -- tips ----------------------------------------------------------------

  async def _unchecked_fw_pick_up_tips(
    self,
    x_positions: List[int],
    y_positions: List[int],
    tip_pattern: List[bool],
    tip_types: List[int],
    begin_z: List[int],
    end_z: List[int],
    traverse_height: List[int],
    height_at_end: List[int],
    pickup_method: List[int],
  ) -> None:
    """Send a tip pickup as it goes on the wire, with every length in tenths of a mm.

    `begin_z` and `end_z` are where the stop disc is: the tip's collar, and `collar_height` below
    it. `traverse_height` and `height_at_end` are in the same frame.
    """
    await self._driver.send_command(
      module="A1PM",
      command="TP",
      xp=x_positions,
      yp=y_positions,
      tm=tip_pattern,
      tt=tip_types,
      tp=begin_z,
      tz=end_z,
      th=traverse_height,
      te=height_at_end,
      ba=[0] * len(x_positions),
      td=pickup_method,
    )

  async def _unchecked_fw_drop_tips(
    self,
    x_positions: List[int],
    y_positions: List[int],
    tip_pattern: List[bool],
    begin_z: List[int],
    end_z: List[int],
    traverse_height: List[int],
    height_at_end: List[int],
    dispose_method: List[int],
  ) -> None:
    """Send a tip drop as it goes on the wire, with every length in tenths of a mm.

    `begin_z` and `end_z` are where the tip's end is, as the machine takes them for a drop.
    """
    await self._driver.send_command(
      module="A1PM",
      command="TR",
      xp=x_positions,
      yp=y_positions,
      tp=begin_z,
      tz=end_z,
      th=traverse_height,
      te=height_at_end,
      tm=tip_pattern,
      ts=0,
      td=dispose_method,
    )

  # -- aspiration and dispensing -------------------------------------------

  async def _unchecked_fw_aspirate(
    self,
    aspiration_type: List[int],
    tip_pattern: List[bool],
    x_positions: List[int],
    y_positions: List[int],
    minimum_traverse_height_start: List[int],
    minimum_z_end_position: List[int],
    lld_search_heights: List[int],
    clot_detection_heights: List[int],
    liquid_surfaces_no_lld: List[int],
    pull_out_distances_transport_air: List[int],
    second_section_heights: List[int],
    second_section_ratios: List[int],
    minimum_heights: List[int],
    immersion_depths: List[int],
    surface_following_distances: List[int],
    aspiration_volumes: List[int],
    aspiration_speeds: List[int],
    transport_air_volumes: List[int],
    blow_out_air_volumes: List[int],
    pre_wetting_volumes: List[int],
    lld_modes: List[int],
    clld_sensitivities: List[int],
    plld_sensitivities: List[int],
    dual_lld_max_differences: List[int],
    aspirate_positions_above_z_touch_off: List[int],
    swap_speeds: List[int],
    settling_times: List[int],
    mix_volumes: List[int],
    mix_cycles: List[int],
    mix_positions_from_liquid_surface: List[int],
    mix_speeds: List[int],
    mix_surface_following_distances: List[int],
    adc_algorithms: List[bool],
    capacitive_mad_supervisions: List[bool],
    pressure_mad_supervisions: List[bool],
    tadm_algorithm: bool,
    limit_curve_indices: List[int],
    recording_mode: int,
  ) -> None:
    """Send an aspiration as it goes on the wire (`A1PM DA`), with every value checked by nothing.

    Every list holds one value for each channel of the machine, and every value is already in the
    unit the wire carries: tenths of a mm, hundredths of a uL for a volume, tenths for a speed in
    uL/s. `PipettesConfiguration` converts from mm, uL and uL/s.

    Args:
      aspiration_type: 0 simple, 1 sequence, 2 cup emptied (`at`).
      tip_pattern: which channels take part (`tm`).
      x_positions: where each channel goes, along X (`xp`).
      y_positions: where each channel goes, along Y (`yp`).
      minimum_traverse_height_start: how high the channels travel first (`th`).
      minimum_z_end_position: how high they end (`te`).
      lld_search_heights: where each channel starts searching for the liquid (`lp`).
      clot_detection_heights: how far below the surface a clot is looked for (`ch`).
      liquid_surfaces_no_lld: the surface each channel uses when LLD is off (`zl`).
      pull_out_distances_transport_air: how far a channel pulls out to take transport air (`po`).
      second_section_heights: the second section's height, measured from the Z reference (`zu`).
      second_section_ratios: the second section's ratio, in ten-thousandths (`zr`).
      minimum_heights: the lowest each channel may go (`zx`).
      immersion_depths: how far into the liquid each tip goes, negative being out of it (`ip`).
      surface_following_distances: how far each channel follows the surface (`fp`).
      aspiration_volumes: what each channel draws (`av`).
      aspiration_speeds: how fast (`as`).
      transport_air_volumes: transport air drawn per channel (`ta`).
      blow_out_air_volumes: air blown out per channel (`ba`).
      pre_wetting_volumes: drawn and returned before the aspiration (`oa`).
      lld_modes: how each channel finds the liquid, 0 off to 4 (`lm`).
      clld_sensitivities: capacitive LLD sensitivity, 1 high to 4 low (`ll`).
      plld_sensitivities: pressure LLD sensitivity, 1 high to 4 low (`lv`).
      dual_lld_max_differences: the most the capacitive and pressure LLD positions may differ in
        a dual search (`ld`).
      aspirate_positions_above_z_touch_off: where a channel draws above a Z touch-off (`zo`).
      swap_speeds: how fast each channel leaves the liquid (`de`).
      settling_times: how long each channel waits after drawing (`wt`).
      mix_volumes: what each channel mixes with (`mv`).
      mix_cycles: how many times each channel mixes (`mc`).
      mix_positions_from_liquid_surface: where each channel mixes, from its actual position
        (`mp`).
      mix_speeds: how fast each channel mixes (`ms`).
      mix_surface_following_distances: how far each channel follows during mixing (`mh`).
      adc_algorithms: whether each channel uses the ADC algorithm (`la`).
      capacitive_mad_supervisions: whether capacitive MAD supervision is on (`lb`).
      pressure_mad_supervisions: whether pressure MAD supervision is on (`lc`).
      tadm_algorithm: whether the TADM algorithm is on (`gj`).
      limit_curve_indices: which limit curve each channel uses (`gi`).
      recording_mode: 0 none, 1 TADM errors, 2 all TADM measurements (`gk`).
    """
    await self._driver.send_command(
      module="A1PM",
      command="DA",
      at=aspiration_type,
      tm=tip_pattern,
      xp=x_positions,
      yp=y_positions,
      th=minimum_traverse_height_start,
      te=minimum_z_end_position,
      lp=lld_search_heights,
      ch=clot_detection_heights,
      zl=liquid_surfaces_no_lld,
      po=pull_out_distances_transport_air,
      zu=second_section_heights,
      zr=second_section_ratios,
      zx=minimum_heights,
      ip=immersion_depths,
      fp=surface_following_distances,
      av=aspiration_volumes,
      as_=aspiration_speeds,
      ta=transport_air_volumes,
      ba=blow_out_air_volumes,
      oa=pre_wetting_volumes,
      lm=lld_modes,
      ll=clld_sensitivities,
      lv=plld_sensitivities,
      ld=dual_lld_max_differences,
      zo=aspirate_positions_above_z_touch_off,
      de=swap_speeds,
      wt=settling_times,
      mv=mix_volumes,
      mc=mix_cycles,
      mp=mix_positions_from_liquid_surface,
      ms=mix_speeds,
      mh=mix_surface_following_distances,
      la=adc_algorithms,
      lb=capacitive_mad_supervisions,
      lc=pressure_mad_supervisions,
      gj=tadm_algorithm,
      gi=limit_curve_indices,
      gk=recording_mode,
    )

  async def _unchecked_fw_dispense(
    self,
    dispense_mode: List[int],
    tip_pattern: List[bool],
    x_positions: List[int],
    y_positions: List[int],
    minimum_heights: List[int],
    lld_search_heights: List[int],
    liquid_surfaces_no_lld: List[int],
    pull_out_distances_transport_air: List[int],
    immersion_depths: List[int],
    surface_following_distances: List[int],
    second_section_heights: List[int],
    second_section_ratios: List[int],
    minimum_traverse_height_start: List[int],
    minimum_z_end_position: List[int],
    dispense_volumes: List[int],
    dispense_speeds: List[int],
    cut_off_speeds: List[int],
    stop_back_volumes: List[int],
    transport_air_volumes: List[int],
    blow_out_air_volumes: List[int],
    lld_modes: List[int],
    side_touch_off_distance: int,
    dispense_positions_above_z_touch_off: List[int],
    clld_sensitivities: List[int],
    plld_sensitivities: List[int],
    swap_speeds: List[int],
    settling_times: List[int],
    mix_volumes: List[int],
    mix_cycles: List[int],
    mix_positions_from_liquid_surface: List[int],
    mix_speeds: List[int],
    mix_surface_following_distances: List[int],
    adc_algorithms: List[bool],
    tadm_algorithm: bool,
    limit_curve_indices: List[int],
    recording_mode: int,
  ) -> None:
    """Send a dispense as it goes on the wire (`A1PM DD`), with every value checked by nothing.

    Every list holds one value for each channel of the machine, and every value is already in the
    unit the wire carries: tenths of a mm, hundredths of a uL for a volume, tenths for a speed in
    uL/s. `PipettesConfiguration` converts from mm, uL and uL/s.

    Args:
      dispense_mode: the mode from `_get_dispense_mode` (`dm`).
      tip_pattern: which channels take part (`tm`).
      x_positions: where each channel goes, along X (`xp`).
      y_positions: where each channel goes, along Y (`yp`).
      minimum_heights: the lowest each channel may go (`zx`).
      lld_search_heights: where each channel starts searching for the liquid (`lp`).
      liquid_surfaces_no_lld: the surface each channel uses when LLD is off (`zl`).
      pull_out_distances_transport_air: how far a channel pulls out to take transport air (`po`).
      immersion_depths: how far into the liquid each tip goes, negative being out of it (`ip`).
      surface_following_distances: how far each channel follows the surface (`fp`).
      second_section_heights: the second section's height, measured from the Z reference (`zu`).
      second_section_ratios: the second section's ratio, in ten-thousandths (`zr`).
      minimum_traverse_height_start: how high the channels travel first (`th`).
      minimum_z_end_position: how high they end (`te`).
      dispense_volumes: what each channel gives (`dv`).
      dispense_speeds: how fast (`ds`).
      cut_off_speeds: the speed each channel cuts off at (`ss`).
      stop_back_volumes: how much each channel stops back (`rv`).
      transport_air_volumes: transport air drawn per channel (`ta`).
      blow_out_air_volumes: air blown out per channel (`ba`).
      lld_modes: how each channel finds the liquid, 0 off to 4 (`lm`).
      side_touch_off_distance: how far a channel touches off to the side (`dj`).
      dispense_positions_above_z_touch_off: where a channel dispenses above a Z touch-off (`zo`).
      clld_sensitivities: capacitive LLD sensitivity, 1 high to 4 low (`ll`).
      plld_sensitivities: pressure LLD sensitivity, 1 high to 4 low (`lv`).
      swap_speeds: how fast each channel leaves the liquid (`de`).
      settling_times: how long each channel waits after dispensing (`wt`).
      mix_volumes: what each channel mixes with (`mv`).
      mix_cycles: how many times each channel mixes (`mc`).
      mix_positions_from_liquid_surface: where each channel mixes, from its actual position
        (`mp`).
      mix_speeds: how fast each channel mixes (`ms`).
      mix_surface_following_distances: how far each channel follows during mixing (`mh`).
      adc_algorithms: whether each channel uses the ADC algorithm (`la`).
      tadm_algorithm: whether the TADM algorithm is on (`gj`).
      limit_curve_indices: which limit curve each channel uses (`gi`).
      recording_mode: 0 none, 1 TADM errors, 2 all TADM measurements (`gk`).
    """
    await self._driver.send_command(
      module="A1PM",
      command="DD",
      dm=dispense_mode,
      tm=tip_pattern,
      xp=x_positions,
      yp=y_positions,
      zx=minimum_heights,
      lp=lld_search_heights,
      zl=liquid_surfaces_no_lld,
      po=pull_out_distances_transport_air,
      ip=immersion_depths,
      fp=surface_following_distances,
      zu=second_section_heights,
      zr=second_section_ratios,
      th=minimum_traverse_height_start,
      te=minimum_z_end_position,
      # At least four digits, as the machine takes the volume: a shorter number is read wrong.
      dv=[f"{volume:04}" for volume in dispense_volumes],
      ds=dispense_speeds,
      ss=cut_off_speeds,
      rv=stop_back_volumes,
      ta=transport_air_volumes,
      ba=blow_out_air_volumes,
      lm=lld_modes,
      dj=side_touch_off_distance,
      zo=dispense_positions_above_z_touch_off,
      ll=clld_sensitivities,
      lv=plld_sensitivities,
      de=swap_speeds,
      wt=settling_times,
      mv=mix_volumes,
      mc=mix_cycles,
      mp=mix_positions_from_liquid_surface,
      ms=mix_speeds,
      mh=mix_surface_following_distances,
      la=adc_algorithms,
      gj=tadm_algorithm,
      gi=limit_curve_indices,
      gk=recording_mode,
    )

  async def pick_up_tips(
    self,
    tip_spots: Sequence[TipSpot],
    use_channels: Optional[List[int]] = None,
    offsets: Optional[List[Coordinate]] = None,
    begin_tip_pick_up_process: Optional[float] = None,
    end_tip_pick_up_process: Optional[float] = None,
    minimum_traverse_height_start: Optional[float] = None,
  ) -> None:
    """Pick up a tip from each spot, one channel per spot. Moves the channels.

    The spots go in one command, so their tips must all be of the same model. Each tip is written
    into the machine's tip type table if it is not there yet, and taken out of its spot in the
    model once it is on its channel.

    Heights as legacy: the process begins a collar's height above the highest spot, and ends at the
    spot. If the machine reports an error, the channels are asked which of them carry a tip, and
    the model takes that answer.

    Args:
      tip_spots: where to pick up, one per channel, in ascending channel order.
      use_channels: which channels, 0-indexed from the back. The first `len(tip_spots)` when None.
      offsets: added to each spot's centre, in mm. None for none.
      begin_tip_pick_up_process: where the pick-up begins, in mm. The spot plus the collar height
        when None.
      end_tip_pick_up_process: where it ends, in mm. The highest spot when None.
      minimum_traverse_height_start: how high the channels travel first, in mm. 245.0 when None.

    Raises:
      RuntimeError: If the driver has no deck.
      ValueError: If the spots and channels do not match, a position is out of reach, or the tips
        are not all of one model.
      NoTipError: If a spot holds no tip while tip tracking is on.
      HasTipError: If a channel already carries a tip.
    """
    if not tip_spots:
      return
    channels, offsets_used = self._channels_and_offsets(tip_spots, use_channels, offsets)
    for channel in channels:
      if channel in self._mounted_tips:
        raise HasTipError(f"channel {channel} already carries {self._mounted_tips[channel].name}")

    tips = self._tips_to_collect(tip_spots)
    xs, ys = self._spot_positions(tip_spots, channels, offsets_used)
    config = self.configuration
    tip_types = [await self._driver.get_or_assign_tip_type_index(tip) for tip in tips]

    deck = self._driver.deck
    assert deck is not None  # checked by `_spot_positions`
    zs = [
      spot.get_location_wrt(deck, z="b").z + offset.z
      for spot, offset in zip(tip_spots, offsets_used)
    ]
    collar = tips[0].collar_height
    end = [config.to_increments(z) for z in zs]
    begin = (
      [config.to_increments(z + collar) for z in zs]
      if begin_tip_pick_up_process is None
      else [config.to_increments(begin_tip_pick_up_process)] * len(zs)
    )
    if end_tip_pick_up_process is not None:
      end = [config.to_increments(end_tip_pick_up_process)] * len(zs)
    _check_range(
      "end_tip_pick_up_process", [z / config.increments_per_mm for z in end], config.z_range
    )
    _check_range(
      "begin_tip_pick_up_process", [z / config.increments_per_mm for z in begin], config.z_range
    )

    travel = (
      self.configuration.default_minimal_traverse_height
      if minimum_traverse_height_start is None
      else minimum_traverse_height_start
    )
    _check_traverse_height(travel, config)
    travel_increments = config.to_increments(travel)
    try:
      await self._unchecked_fw_pick_up_tips(
        x_positions=self._full_width(channels, xs),
        y_positions=self._full_width(channels, ys),
        tip_pattern=self._tip_pattern(channels),
        tip_types=self._full_width(channels, tip_types),
        begin_z=self._full_width(channels, begin),
        end_z=self._full_width(channels, end),
        traverse_height=self._full_width(channels, [travel_increments] * len(xs)),
        height_at_end=self._full_width(channels, [travel_increments] * len(xs)),
        pickup_method=self._full_width(channels, [tip.pickup_method.value + 1 for tip in tips]),
      )
    except BaseException:
      # A command can stop part way. The channels themselves say which of them came away with a
      # tip, which is worth more than the plan.
      await self._record_after_failed_tip_command(channels, tip_spots, tips)
      raise

    for channel, spot, tip in zip(channels, tip_spots, tips):
      if spot.tip is not None:
        spot.unassign_tip()
      self._mounted_tips[channel] = tip

  async def drop_tips(
    self,
    destinations: Sequence[TipSpot],
    use_channels: Optional[List[int]] = None,
    offsets: Optional[List[Coordinate]] = None,
    begin_tip_deposit_process: Optional[float] = None,
    end_tip_deposit_process: Optional[float] = None,
    minimum_traverse_height_start: Optional[float] = None,
  ) -> None:
    """Put the tip each channel carries back into a spot. Moves the channels.

    The spots go in one command. As legacy, the process ends where the tip's end sits in the spot,
    which is the spot less the length of the tip that hangs below its collar, and begins 10 mm
    above that. If the machine reports an error, the channels are asked which of them carry a tip,
    and the model takes that answer.

    Args:
      destinations: where to put the tips, one per channel, in ascending channel order.
      use_channels: which channels, 0-indexed from the back. The first `len(tip_spots)` when None.
      offsets: added to each spot's centre, in mm. None for none.
      begin_tip_deposit_process: where the drop begins, in mm. The end plus 10 mm when None.
      end_tip_deposit_process: where it ends, in mm. Where the tip's end sits when None.
      minimum_traverse_height_start: how high the channels travel first, in mm. 245.0 when None.

    Raises:
      RuntimeError: If the driver has no deck.
      ValueError: If the spots and channels do not match, or a position is out of reach.
      NoTipError: If a channel carries no tip in the model.
      HasTipError: If a spot already holds a tip while tip tracking is on.
    """
    if not destinations:
      return
    channels, offsets_used = self._channels_and_offsets(destinations, use_channels, offsets)
    for channel in channels:
      if channel not in self._mounted_tips:
        raise NoTipError(f"channel {channel} carries no tip")
    for spot in destinations:
      if spot.tip is not None:
        raise HasTipError(f"{spot.name} already holds a tip")

    xs, ys = self._spot_positions(destinations, channels, offsets_used)
    config = self.configuration
    deck = self._driver.deck
    assert deck is not None  # checked by `_spot_positions`
    ends = []
    for spot, channel, offset in zip(destinations, channels, offsets_used):
      tip = self._mounted_tips[channel]
      seat = spot.get_location_wrt(deck, z="b").z + offset.z
      ends.append(seat - (tip.get_size_z() - tip.collar_height))
    end = (
      [config.to_increments(z) for z in ends]
      if end_tip_deposit_process is None
      else [config.to_increments(end_tip_deposit_process)] * len(ends)
    )
    begin = (
      [z + config.to_increments(10.0) for z in end]
      if begin_tip_deposit_process is None
      else [config.to_increments(begin_tip_deposit_process)] * len(ends)
    )
    _check_range(
      "end_tip_deposit_process", [z / config.increments_per_mm for z in end], config.z_range
    )
    _check_range(
      "begin_tip_deposit_process", [z / config.increments_per_mm for z in begin], config.z_range
    )

    travel = (
      self.configuration.default_minimal_traverse_height
      if minimum_traverse_height_start is None
      else minimum_traverse_height_start
    )
    _check_traverse_height(travel, config)
    travel_increments = config.to_increments(travel)

    try:
      await self._unchecked_fw_drop_tips(
        x_positions=self._full_width(channels, xs),
        y_positions=self._full_width(channels, ys),
        tip_pattern=self._tip_pattern(channels),
        begin_z=self._full_width(channels, begin),
        end_z=self._full_width(channels, end),
        traverse_height=self._full_width(channels, [travel_increments] * len(xs)),
        height_at_end=self._full_width(channels, [travel_increments] * len(xs)),
        dispose_method=self._full_width(channels, [0] * len(xs)),
      )
    except BaseException:
      self._warn_after_failed_drop(channels)
      raise

    for channel, spot in zip(channels, destinations):
      tip = self._mounted_tips.pop(channel)
      spot.assign_tip(tip)

  async def _record_after_failed_tip_command(
    self,
    channels: List[int],
    tip_spots: Sequence[TipSpot],
    tips: Sequence[HamiltonTip],
  ) -> None:
    """Take the machine's word for which tips came up, after a pickup failed.

    A pickup can stop part way. A channel that carries a tip now took it out of its spot, and that
    is recorded; a channel that does not was left as it was, with its tip still in the spot.

    Args:
      channels: the channels the failed command was about.
      tip_spots: their spots, in the same order.
      tips: what each was to collect, in the same order.
    """
    presence = await self.sense_tip_presence()
    for channel, spot, tip in zip(channels, tip_spots, tips):
      if presence[channel] and channel not in self._mounted_tips:
        if spot.tip is not None:
          spot.unassign_tip()
        self._mounted_tips[channel] = tip

  def _warn_after_failed_drop(self, channels: List[int]) -> None:
    """Say that a failed drop leaves the model guessing which tips went back into their spots."""
    logger.warning(
      "a drop that failed leaves the model unsure where the tips on channels %s are: they are "
      "not recorded as back in their spots",
      channels,
    )
