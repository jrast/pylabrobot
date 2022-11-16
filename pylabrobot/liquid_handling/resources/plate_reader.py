""" Defines PlateReader class. """

from typing import Optional

from pylabrobot.liquid_handling.resources.abstract import Coordinate, Resource


class PlateReader(Resource):
  """ A base for plate readers. """

  def __init__(self, name: str):
    super().__init__(name=name, size_x=0, size_y=0, size_z=0, category="plate_reader")

  def assign_child_resource(self, resource):
    if len(self.children) >= 1:
      raise ValueError("There already is a plate in the plate reader.")
    super().assign_child_resource(resource, location=Coordinate.zero())

  def get_plate(self) -> Optional[Resource]:
    if len(self.children) == 0:
      return None
    return self.children[0]
