from typing import Optional
from pylabrobot.resources.plate import Lid, Plate
from pylabrobot.resources.well import (
  Well,
  WellBottomType,
  CrossSectionType,
)
from pylabrobot.resources.utils import create_ordered_items_2d

from pylabrobot.resources.height_volume_functions import (
  calculate_liquid_height_container_1segment_round_fbottom,
  calculate_liquid_height_in_container_2segments_square_vbottom,
  calculate_liquid_volume_container_1segment_round_fbottom,
  calculate_liquid_volume_container_2segments_square_vbottom,
  compute_height_from_volume_conical_frustum,
  compute_volume_from_height_conical_frustum,
)


def Cos_96_EZWash(name: str, with_lid: bool = False) -> Plate:
  raise ValueError("Deprecated. You probably want to use Cor_96_wellplate_360ul_Fb instead.")


def Cos_6_wellplate_16800ul_Fb_Lid(name: str) -> Lid:
  return Lid(
    name=name,
    size_x=127.0,
    size_y=86.0,
    size_z=7.8,
    nesting_z_height=6.7,  # measure overlap between lid and plate
    model="Cos_6_wellplate_16800ul_Fb_Lid",
  )


def _compute_volume_from_height_Cos_6_wellplate_16800ul_Fb(h: float):
  if h > 18.0:
    raise ValueError(f"Height {h} is too large for Cos_6_wellplate_16800ul_Fb")
  return calculate_liquid_volume_container_1segment_round_fbottom(
    d=35.0, h_cylinder=18.2, liquid_height=h
  )


def _compute_height_from_volume_Cos_6_wellplate_16800ul_Fb(
  liquid_volume: float,
):
  if liquid_volume > 17_640:  # 5% tolerance
    raise ValueError(f"Volume {liquid_volume} is too large for Cos_6_wellplate_16800ul_Fb")
  return calculate_liquid_height_container_1segment_round_fbottom(
    d=35.0, h_cylinder=18.2, liquid_volume=liquid_volume
  )


def Cos_6_wellplate_16800ul_Fb(name: str, with_lid: bool = True) -> Plate:
  """Corning-Costar 6-well multi-well plate (MWP); product no.: 3516.
  - Material: ?
  - Cleanliness: 3516: sterilized by gamma irradiation
  - Nonreversible lids with condensation rings to reduce contamination
  - Treated for optimal cell attachment
  - Cell growth area: 9.5 cm² (approx.)
  - Total volume: 16.8 mL
  """
  return Plate(
    name=name,
    size_x=127.0,
    size_y=86.0,
    size_z=20.0,
    lid=Cos_6_wellplate_16800ul_Fb_Lid(name=name + "_lid") if with_lid else None,
    model="Cos_6_wellplate_16800ul_Fb",
    ordered_items=create_ordered_items_2d(
      Well,
      num_items_x=3,
      num_items_y=2,
      dx=7.0,
      dy=5.45,
      dz=1.35,
      item_dx=38.45,
      item_dy=38.45,
      size_x=35.0,
      size_y=35.0,
      size_z=17.5,
      bottom_type=WellBottomType.FLAT,
      material_z_thickness=1.4,
      cross_section_type=CrossSectionType.CIRCLE,
      compute_volume_from_height=_compute_volume_from_height_Cos_6_wellplate_16800ul_Fb,
      compute_height_from_volume=_compute_height_from_volume_Cos_6_wellplate_16800ul_Fb,
    ),
  )


def Cos_6_wellplate_16800ul_Fb_L(name: str, with_lid: bool = True) -> Plate:
  # https://github.com/PyLabRobot/pylabrobot/issues/252
  raise NotImplementedError(
    "_L and _P definitions are deprecated. Use " "Cos_6_wellplate_16800ul_Fb instead."
  )


def Cos_6_wellplate_16800ul_Fb_P(name: str, with_lid: bool = True) -> Plate:
  # https://github.com/PyLabRobot/pylabrobot/issues/252
  raise NotImplementedError(
    "_L and _P definitions are deprecated. Use "
    "Cos_6_wellplate_16800ul_Fb().rotated(z=90) instead."
  )


# # # # # # # # # # Cos_96_wellplate_2mL_Vb # # # # # # # # # #


def _compute_volume_from_height_Cos_96_wellplate_2mL_Vb(
  h: float,
) -> float:
  if h > 44.1:  # 5% tolerance
    raise ValueError(f"Height {h} is too large for Cos_96_wellplate_2mL_Vb")
  return calculate_liquid_volume_container_2segments_square_vbottom(
    x=7.8, y=7.8, h_pyramid=4.0, h_cube=38.0, liquid_height=h
  )


def _compute_height_from_volume_Cos_96_wellplate_2mL_Vb(
  liquid_volume: float,
):
  if liquid_volume > 2_100:  # 5% tolerance
    raise ValueError(f"Volume {liquid_volume} is too large for Cos_96_wellpate_2mL_Vb")
  return round(
    calculate_liquid_height_in_container_2segments_square_vbottom(
      x=7.8,
      y=7.8,
      h_pyramid=4.0,
      h_cube=38.0,
      liquid_volume=liquid_volume,
    ),
    3,
  )


def Cos_96_wellplate_2mL_Vb_Lid(name: str) -> Lid:
  raise NotImplementedError("This lid is not currently defined.")
  # See https://github.com/PyLabRobot/pylabrobot/pull/161.
  # return Lid(
  #   name=name,
  #   size_x=127.0,
  #   size_y=86.0,
  #   size_z=None,           # measure the total z height
  #   nesting_z_height=None, # measure overlap between lid and plate
  #   model="Cos_96_wellplate_2mL_Vb_Lid",
  # )


def Cos_96_DWP_2mL_Vb(name: str, with_lid: bool = False) -> Plate:
  raise NotImplementedError(
    "This function is deprecated and will be removed in a future version."
    " Use 'Cos_96_wellplate_2mL_Vb' instead."
  )


def Cos_96_wellplate_2mL_Vb(name: str, with_lid: bool = False) -> Plate:
  """Corning 96 deep-well 2 mL PCR plate. Corning cat. no.: 3960
  - Material: Polypropylene
  - Resistant to many common organic solvents (e.g., DMSO, ethanol, methanol)
  - 3960: Sterile and DNase- and RNase-free
  - Total volume: 2 mL
  - Features uniform skirt heights for greater robotic gripping surface
  """
  return Plate(
    name=name,
    size_x=127.0,
    size_y=86.0,
    size_z=43.5,
    lid=Cos_96_wellplate_2mL_Vb_Lid(name=name + "_lid") if with_lid else None,
    model="Cos_96_wellplate_2mL_Vb",
    ordered_items=create_ordered_items_2d(
      Well,
      num_items_x=12,
      num_items_y=8,
      dx=9.6,
      dy=7.0,
      dz=1.2,
      item_dx=9.0,
      item_dy=9.0,
      size_x=8.0,
      size_y=8.0,
      size_z=42.0,
      bottom_type=WellBottomType.V,
      material_z_thickness=1.25,
      cross_section_type=CrossSectionType.RECTANGLE,
      compute_volume_from_height=_compute_volume_from_height_Cos_96_wellplate_2mL_Vb,
      compute_height_from_volume=_compute_height_from_volume_Cos_96_wellplate_2mL_Vb,
    ),
  )


def Cos_96_wellplate_2mL_Vb_L(name: str, with_lid: bool = False) -> Plate:
  # https://github.com/PyLabRobot/pylabrobot/issues/252
  raise NotImplementedError(
    "_L and _P definitions are deprecated. Use " "Cos_96_wellplate_2mL_Vb instead."
  )


def Cos_96_wellplate_2mL_Vb_P(name: str, with_lid: bool = False) -> Plate:
  # https://github.com/PyLabRobot/pylabrobot/issues/252
  raise NotImplementedError(
    "_L and _P definitions are deprecated. Use " "Cos_96_wellplate_2mL_Vb().rotated(z=90) instead."
  )


# # # # # # # # # # Cor_96_wellplate_360ul_Fb # # # # # # # # # #


def Cor_96_wellplate_360ul_Fb_Lid(name: str) -> Lid:
  return Lid(
    name=name,
    size_x=127.76,
    size_y=85.48,
    size_z=8.9,  # measure the total z height
    nesting_z_height=7.6,  # measure overlap between lid and plate
    model="Cor_96_wellplate_360ul_Fb_Lid",
  )


def Cor_96_wellplate_360ul_Fb(name: str, with_lid: bool = False) -> Plate:
  """Cor_96_wellplate_360ul_Fb

  Catalog number 3603

  https://ecatalog.corning.com/life-sciences/b2b/NL/en/Microplates/Assay-Microplates/96-Well-
  Microplates/Corning®-96-well-Black-Clear-and-White-Clear-Bottom-Polystyrene-Microplates/p/3603

  Measurements found here:
  https://www.corning.com/catalog/cls/documents/drawings/MicroplateDimensions96-384-1536.pdf
  https://archive.vn/CnRgl
  """

  # This used to be Cos_96_EZWash in the Esvelt lab
  #
  # return Plate(
  #   name=name,
  #   size_x=127,
  #   size_y=86,
  #   size_z=14.5,
  #   lid=Cos_96_EZWash_Lid(name=name + "_lid") if with_lid else None,
  #   model="Cos_96_EZWash",
  #   ordered_items=create_ordered_items_2d(Well,
  #     num_items_x=12,
  #     num_items_y=8,
  #     dx=10.55,
  #     dy=8.05,
  #     dz=1.0,
  #     item_dx=9.0,
  #     item_dy=9.0,
  #     size_x=6.9,
  #     size_y=6.9,
  #     size_z=11.3,
  #     bottom_type=WellBottomType.FLAT,
  #     cross_section_type=CrossSectionType.CIRCLE,
  #   ),
  # )

  return Plate(
    name=name,
    size_x=127.76,
    size_y=85.48,
    size_z=14.2,
    lid=Cor_96_wellplate_360ul_Fb_Lid(name=name + "_lid") if with_lid else None,
    model="Cor_96_wellplate_360ul_Fb",
    ordered_items=create_ordered_items_2d(
      Well,
      num_items_x=12,
      num_items_y=8,
      dx=10.87,  # 14.3-6.86/2
      dy=7.77,  # 11.2-6.86/2
      dz=3.03,
      item_dx=9.0,
      item_dy=9.0,
      size_x=6.86,  # top
      size_y=6.86,  # top
      size_z=10.67,
      material_z_thickness=0.5,
      bottom_type=WellBottomType.FLAT,
      cross_section_type=CrossSectionType.CIRCLE,
      max_volume=360,
    ),
  )


# # # # # # # # # # Cor_6_wellplate_Fl # # # # # # # # # #


def Cor_6_wellplate_Fl(name: str, lid: Optional[Lid] = None) -> Plate:
  """
  Corning cat. no.: 3471
  - Material: Polystyrene
  - Tissue culture treated: No
  """
  BOTTOM_INNER_WELL_RADIUS = 34.798 / 2  # from Corning Product Description
  TOP_INNER_WELL_RADIUS = 35.433 / 2  # from Corning Product Description

  well_kwargs = {
    "size_x": BOTTOM_INNER_WELL_RADIUS * 2,
    "size_y": BOTTOM_INNER_WELL_RADIUS * 2,
    "size_z": 17.399,  # from Corning Product Description
    "bottom_type": WellBottomType.FLAT,
    "max_volume": 16.850,  # calculated
    "compute_volume_from_height": lambda liquid_height: compute_volume_from_height_conical_frustum(
      liquid_height, BOTTOM_INNER_WELL_RADIUS, TOP_INNER_WELL_RADIUS
    ),
    "compute_height_from_volume": lambda liquid_volume: compute_height_from_volume_conical_frustum(
      liquid_volume, BOTTOM_INNER_WELL_RADIUS, TOP_INNER_WELL_RADIUS
    ),
  }

  return Plate(
    name=name,
    size_x=127.762,  # from Corning Product Description
    size_y=85.471,  # from Corning Product Description
    size_z=22.6314,  # from Corning Product Description
    lid=lid,
    model=Cor_6_wellplate_Fl.__name__,
    ordered_items=create_ordered_items_2d(
      Well,
      num_items_x=3,
      num_items_y=2,
      dx=7.5,  # measured
      dy=6.3,  # measured
      dz=1.8,  # calibrated manually by z-stepping down using a pipette.
      item_dx=39,  # measured
      item_dy=39,  # measured
      **well_kwargs,
    ),
  )
