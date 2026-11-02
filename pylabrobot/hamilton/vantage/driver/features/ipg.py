"""The Vantage's internal plate gripper (IPG), which is the module `A1RM`."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
  from pylabrobot.hamilton.vantage.driver.master import VantageDriver


@dataclass
class IpgConfiguration:
  """What the IPG is asked, and how the machine answers."""

  increments_per_mm: int = 10
  """The wire carries lengths in tenths of a millimeter."""

  x_range: Tuple[float, float] = (-5000.0, 5000.0)
  """Along X, in mm."""

  y_range: Tuple[float, float] = (-1000.0, 1000.0)
  """Along Y, in mm."""

  z_range: Tuple[float, float] = (0.0, 400.0)
  """Along Z, in mm: where the IPG stands, how high it travels, and where a command ends."""

  grip_strength_range: Tuple[int, int] = (0, 160)
  """How hard the gripper closes, from 0 to 160, in 0.1 N."""

  plate_width_range: Tuple[float, float] = (0.0, 999.9)
  """The width of a plate, of the gripper's opening and of its grip, in mm."""

  plate_width_tolerance_range: Tuple[float, float] = (0.0, 9.9)
  """How far a plate's width may be from the width given, in mm."""

  acceleration_index_range: Tuple[int, int] = (0, 4)
  """The acceleration index a move takes."""

  z_clearance_height_range: Tuple[float, float] = (0.0, 99.9)
  """How far above the place a move clears an obstacle, in mm."""

  hotel_depth_range: Tuple[float, float] = (0.0, 300.0)
  """How deep a plate sits in a hotel, in mm."""

  press_on_distance_range: Tuple[float, float] = (0.0, 99.9)
  """How far a put presses on, in mm."""

  move_read_timeout: int = 300
  """How long to wait for a move, in seconds. The arm itself waits that long for a module."""

  default_minimal_traverse_height: float = 245.0
  """How high a move travels when nothing else is asked, in mm."""

  default_minimal_height_at_command_end: float = 245.0
  """How high a grip or a put ends, at least, when nothing else is asked, in mm."""

  default_grip_orientation: int = 32
  """Which way the gripper grips when nothing else is asked: two digits, 1 to 8 each."""

  default_grip_strength: int = 81
  """How hard the gripper closes when nothing else is asked, in 0.1 N."""

  default_acceleration_index: int = 4
  """The acceleration index a grip takes when nothing else is asked."""

  default_press_on_distance: float = 0.5
  """How far a put presses on when nothing else is asked, in mm."""

  open_width_clearance: float = 3.2
  """How much wider than the plate the gripper opens, in mm."""

  grip_width_clearance: float = 3.3
  """How much narrower than the plate the gripper closes, in mm."""

  def to_increments(self, mm: float) -> int:
    """A length in mm as the wire carries it."""
    return round(mm * self.increments_per_mm)


def _check_range(name: str, value: float, limits: Tuple[float, float]) -> None:
  """Raise unless a value is within the limits, both ends included.

  Raises:
    ValueError: If the value is outside them.
  """
  low, high = limits
  if not low <= value <= high:
    raise ValueError(f"{name} must be from {low} to {high}, but is {value}")


def _check_grip_orientation(grip_orientation: int, highest_digit: int) -> None:
  """Raise unless a grip orientation is two digits, each from 1 to `highest_digit`.

  The machine writes one as two digits, the tens and the ones each saying a direction, so 19 is
  not an orientation even though 11 to 88 would let it through.

  Raises:
    ValueError: If either digit is out of range.
  """
  tens, ones = divmod(grip_orientation, 10)
  if not (1 <= tens <= highest_digit and 1 <= ones <= highest_digit):
    raise ValueError(
      f"grip_orientation must be two digits from 1 to {highest_digit}, but is {grip_orientation}"
    )


@dataclass
class IpgParkConfiguration:
  """Where the IPG parks."""

  y: float
  """Along Y, in mm."""

  z: float
  """Along Z, in mm."""

  gripper_width: float
  """How wide the gripper stands, in mm."""

  arm_angle: int
  """The angle of the arm, in degrees."""

  hand_angle: int
  """The angle of the hand, relative to the arm, in degrees."""


class Ipg:
  """The internal plate gripper.

  Reached as `driver.ipg`, which exists only on a machine that has one. Everything here is a
  question to the IPG, except the moves: `initialize`, `park`, `prepare_orientation`, `move_to`,
  `grip_plate` and `put_plate`.

  The machine's grip and put commands carry no traverse height and start where the IPG stands, and
  they need the wrist turned to the grip orientation. So a grip or a put is refused until
  `prepare_orientation` has run since the IPG was initialized or parked, and each one travels to its
  place with a `DN` first.
  """

  def __init__(self, driver: "VantageDriver"):
    self._driver = driver
    self.configuration = IpgConfiguration()

    # Whether the wrist has been turned to a grip orientation since the IPG was initialized or
    # parked. The machine refuses a grip or a put over the command sequence otherwise.
    self._orientation_prepared = False

  async def request_initialization_status(self) -> bool:
    """Whether the IPG is initialized."""
    resp = await self._driver.send_command(module="A1RM", command="QW", fmt={"qw": "int"})
    return bool(resp["qw"] == 1)

  async def request_parking_status(self) -> bool:
    """Whether the IPG is in its park position."""
    resp = await self._driver.send_command(module="A1RM", command="RG", fmt={"rg": "int"})
    return bool(resp["rg"] == 1)

  async def request_plate_presence(self) -> bool:
    """Whether the IPG holds a plate."""
    resp = await self._driver.send_command(module="A1RM", command="QA", fmt={"rt": "int"})
    return bool(resp["rt"] == 1)

  async def request_angles(self) -> List[int]:
    """The angle of the arm and of the hand, relative to the arm, in degrees."""
    resp = await self._driver.send_command(module="A1RM", command="RR", fmt={"rr": "[int]"})
    return list(resp["rr"])

  async def request_park_configuration(self) -> IpgParkConfiguration:
    """Where the IPG parks."""
    resp = await self._driver.send_command(
      module="A1RM", command="RS", fmt={"rs": "[int]", "rr": "[int]"}
    )
    y, z, width = (value / self.configuration.increments_per_mm for value in resp["rs"])
    arm_angle, hand_angle = resp["rr"]
    return IpgParkConfiguration(
      y=y, z=z, gripper_width=width, arm_angle=arm_angle, hand_angle=hand_angle
    )

  async def request_access_range(self, grip_orientation: Optional[int] = None) -> str:
    """Ask how far the IPG can reach at a grip orientation.

    Args:
      grip_orientation: which way the gripper would grip, as two digits from 1 to 8 each (11 to
        88). The configured default when None.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.
    """
    orientation = (
      self.configuration.default_grip_orientation if grip_orientation is None else grip_orientation
    )
    _check_grip_orientation(orientation, highest_digit=8)
    return await self._driver.send_command(module="A1RM", command="QR", gd=orientation)

  async def request_position(self, grip_orientation: Optional[int] = None) -> str:
    """Ask where the IPG would stand to grip at an orientation.

    The orientation takes a digit of 9, unlike `request_access_range`.

    Args:
      grip_orientation: which way the gripper would grip, as two digits from 1 to 9 each (11 to
        99). The configured default when None.

    Returns:
      The machine's reply as it writes it, which is not read into anything yet.
    """
    orientation = (
      self.configuration.default_grip_orientation if grip_orientation is None else grip_orientation
    )
    _check_grip_orientation(orientation, highest_digit=9)
    return await self._driver.send_command(module="A1RM", command="QI", gd=orientation)

  async def initialize(self) -> None:
    """Initialize all the drives of the IPG. Moves it."""
    await self._driver.send_command(
      module="A1RM", command="DI", read_timeout=self.configuration.move_read_timeout
    )
    self._orientation_prepared = False

  async def park(self) -> None:
    """Move the IPG to its park position. Moves it."""
    await self._driver.send_command(
      module="A1RM", command="GP", read_timeout=self.configuration.move_read_timeout
    )
    self._orientation_prepared = False

  def _traverse_height(self, height: Optional[float]) -> int:
    """A height to travel at, in increments, checked against what the IPG reaches.

    Raises:
      ValueError: If the height is out of range.
    """
    value = self.configuration.default_minimal_traverse_height if height is None else height
    _check_range("minimal_traverse_height", value, self.configuration.z_range)
    return self.configuration.to_increments(value)

  async def _move_to(
    self,
    x: float,
    y: float,
    z: float,
    minimal_traverse_height: Optional[float],
  ) -> None:
    """Move to a place, travelling at a height that clears the deck.

    Raises:
      ValueError: If a coordinate or the height is out of range.
    """
    _check_range("x", x, self.configuration.x_range)
    _check_range("y", y, self.configuration.y_range)
    _check_range("z", z, self.configuration.z_range)
    await self._driver.send_command(
      module="A1RM",
      command="DN",
      xp=self.configuration.to_increments(x),
      yp=self.configuration.to_increments(y),
      zp=self.configuration.to_increments(z),
      th=self._traverse_height(minimal_traverse_height),
      read_timeout=self.configuration.move_read_timeout,
    )

  def _require_orientation(self, instead: str) -> None:
    """Raise unless the wrist has been turned to a grip orientation since the last initialization.

    Raises:
      ValueError: If `prepare_orientation` has not run since then, naming the method to reach for.
    """
    if not self._orientation_prepared:
      raise ValueError(
        f"the gripper's orientation is not prepared, so `{instead}` is refused: the machine's grip "
        "and put commands start where the IPG stands and need the wrist turned to a grip "
        "orientation, so call `prepare_orientation` first"
      )

  async def prepare_orientation(
    self,
    grip_orientation: Optional[int] = None,
    minimal_traverse_height: Optional[float] = None,
  ) -> None:
    """Turn the wrist to the grip orientation, travelling at a height that clears the deck.

    A grip or a put has to have this before it.

    Args:
      grip_orientation: which way to grip, as two digits from 1 to 8 each (11 to 88). The
        configured default when None.
      minimal_traverse_height: how high the IPG travels first, in mm. The configured default when
        None.

    Raises:
      ValueError: If the orientation or the height is out of range.
    """
    orientation = (
      self.configuration.default_grip_orientation if grip_orientation is None else grip_orientation
    )
    _check_grip_orientation(orientation, highest_digit=8)
    await self._driver.send_command(
      module="A1RM",
      command="GA",
      gd=orientation,
      th=self._traverse_height(minimal_traverse_height),
      read_timeout=self.configuration.move_read_timeout,
    )
    self._orientation_prepared = True

  async def move_to(
    self,
    x: float,
    y: float,
    z: float,
    minimal_traverse_height: Optional[float] = None,
  ) -> None:
    """Move the IPG to a place, travelling at a height that clears the deck. Moves it.

    Args:
      x: along X, in mm.
      y: along Y, in mm.
      z: along Z, in mm.
      minimal_traverse_height: how high the IPG travels first, in mm. The configured default when
        None.

    Raises:
      ValueError: If a coordinate or the height is out of range.
    """
    await self._move_to(x, y, z, minimal_traverse_height)

  async def grip_plate(
    self,
    x: float,
    y: float,
    z: float,
    *,
    plate_width: float,
    open_gripper_width: Optional[float] = None,
    grip_width: Optional[float] = None,
    grip_strength: Optional[int] = None,
    plate_width_tolerance: float = 2.0,
    acceleration_index: Optional[int] = None,
    z_clearance_height: float = 0.0,
    hotel_depth: float = 0.0,
    minimal_traverse_height: Optional[float] = None,
    minimal_height_at_command_end: Optional[float] = None,
  ) -> None:
    """Grip a plate at a place. Moves the IPG.

    The IPG travels to the place with a `DN`, and then grips with a `DG`, which starts where it
    stands. `prepare_orientation` has to have run since it was initialized or parked.

    Args:
      x: where the plate's centre is, along X, in mm.
      y: where its centre is, along Y, in mm.
      z: the height it is gripped at, along Z, in mm.
      plate_width: how wide the plate is, in mm.
      open_gripper_width: how wide the gripper opens before it closes, in mm. The plate's width
        plus the configured clearance when None.
      grip_width: how narrow the gripper closes on the plate, in mm. The plate's width less the
        configured clearance when None.
      grip_strength: how hard it closes, from 0 to 160, in 0.1 N. The configured default when
        None.
      plate_width_tolerance: how far the plate's width may be from `plate_width`, in mm.
      acceleration_index: from 0 to 4. The configured default when None.
      z_clearance_height: how far above the grip height the gripper clears an obstacle, in mm.
      hotel_depth: how deep the plate sits in a hotel, in mm, 0 being a stack.
      minimal_traverse_height: how high the IPG travels first, in mm. The configured default when
        None.
      minimal_height_at_command_end: how high the command ends, in mm, at least. The configured
        default when None.

    Raises:
      ValueError: If the orientation is not prepared, or a value is out of range.
    """
    self._require_orientation("grip_plate")
    open_width = (
      plate_width + self.configuration.open_width_clearance
      if open_gripper_width is None
      else open_gripper_width
    )
    closed_width = (
      plate_width - self.configuration.grip_width_clearance if grip_width is None else grip_width
    )
    strength = self.configuration.default_grip_strength if grip_strength is None else grip_strength
    acceleration = (
      self.configuration.default_acceleration_index
      if acceleration_index is None
      else acceleration_index
    )
    end_height = (
      self.configuration.default_minimal_height_at_command_end
      if minimal_height_at_command_end is None
      else minimal_height_at_command_end
    )
    _check_range("plate_width", plate_width, self.configuration.plate_width_range)
    _check_range("open_gripper_width", open_width, self.configuration.plate_width_range)
    _check_range("grip_width", closed_width, self.configuration.plate_width_range)
    _check_range("grip_strength", strength, self.configuration.grip_strength_range)
    _check_range(
      "plate_width_tolerance", plate_width_tolerance, self.configuration.plate_width_tolerance_range
    )
    _check_range("acceleration_index", acceleration, self.configuration.acceleration_index_range)
    _check_range(
      "z_clearance_height", z_clearance_height, self.configuration.z_clearance_height_range
    )
    _check_range("hotel_depth", hotel_depth, self.configuration.hotel_depth_range)
    _check_range("minimal_height_at_command_end", end_height, self.configuration.z_range)

    await self._move_to(x, y, z, minimal_traverse_height)
    await self._driver.send_command(
      module="A1RM",
      command="DG",
      xp=self.configuration.to_increments(x),
      yp=self.configuration.to_increments(y),
      zp=self.configuration.to_increments(z),
      yw=strength,
      yo=self.configuration.to_increments(open_width),
      yg=self.configuration.to_increments(closed_width),
      pt=self.configuration.to_increments(plate_width_tolerance),
      ai=acceleration,
      zc=self.configuration.to_increments(z_clearance_height),
      hd=self.configuration.to_increments(hotel_depth),
      te=self.configuration.to_increments(end_height),
      read_timeout=self.configuration.move_read_timeout,
    )

  async def put_plate(
    self,
    x: float,
    y: float,
    z: float,
    *,
    plate_width: float,
    open_gripper_width: Optional[float] = None,
    z_clearance_height: float = 0.0,
    press_on_distance: Optional[float] = None,
    hotel_depth: float = 0.0,
    minimal_traverse_height: Optional[float] = None,
    minimal_height_at_command_end: Optional[float] = None,
  ) -> None:
    """Put the plate the IPG carries down at a place. Moves the IPG.

    The IPG travels to the place with a `DN`, and then puts the plate down with a `DR`, which
    starts where it stands. `prepare_orientation` has to have run since it was initialized or
    parked.

    Args:
      x: where the plate goes, along X, in mm.
      y: where it goes, along Y, in mm.
      z: the height it is let go at, along Z, in mm.
      plate_width: how wide the plate is, in mm.
      open_gripper_width: how wide the gripper opens to let go of it, in mm. The plate's width
        plus the configured clearance when None.
      z_clearance_height: how far above the place the gripper clears an obstacle, in mm.
      press_on_distance: how far it presses the plate on, in mm. The configured default when None.
      hotel_depth: how deep the plate sits in a hotel, in mm, 0 being a stack.
      minimal_traverse_height: how high the IPG travels first, in mm. The configured default when
        None.
      minimal_height_at_command_end: how high the command ends, in mm, at least. The configured
        default when None.

    Raises:
      ValueError: If the orientation is not prepared, or a value is out of range.
    """
    self._require_orientation("put_plate")
    open_width = (
      plate_width + self.configuration.open_width_clearance
      if open_gripper_width is None
      else open_gripper_width
    )
    press_on = (
      self.configuration.default_press_on_distance
      if press_on_distance is None
      else press_on_distance
    )
    end_height = (
      self.configuration.default_minimal_height_at_command_end
      if minimal_height_at_command_end is None
      else minimal_height_at_command_end
    )
    _check_range("plate_width", plate_width, self.configuration.plate_width_range)
    _check_range("open_gripper_width", open_width, self.configuration.plate_width_range)
    _check_range(
      "z_clearance_height", z_clearance_height, self.configuration.z_clearance_height_range
    )
    _check_range("press_on_distance", press_on, self.configuration.press_on_distance_range)
    _check_range("hotel_depth", hotel_depth, self.configuration.hotel_depth_range)
    _check_range("minimal_height_at_command_end", end_height, self.configuration.z_range)

    await self._move_to(x, y, z, minimal_traverse_height)
    await self._driver.send_command(
      module="A1RM",
      command="DR",
      xp=self.configuration.to_increments(x),
      yp=self.configuration.to_increments(y),
      zp=self.configuration.to_increments(z),
      yo=self.configuration.to_increments(open_width),
      zc=self.configuration.to_increments(z_clearance_height),
      zi=self.configuration.to_increments(press_on),
      hd=self.configuration.to_increments(hotel_depth),
      te=self.configuration.to_increments(end_height),
      read_timeout=self.configuration.move_read_timeout,
    )
