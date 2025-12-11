from ..base import Entry
from ....room import Room, get_material
from ....core.constants import DEFAULT_SAMPLING_RATE
import math

ROOM_ASPECT_RATIOS = {
    4.5: (1.5, 1.0),
    6: (1.5, 1.0),
    8: (1.6, 1.0),
    9.5: (1.7, 1.0),
    12: (1.8, 1.0),
    15: (2.0, 1.0),
    16: (2.0, 1.0),
    18: (2.2, 1.0),
    20: (2.5, 1.0),
    24: (2.4, 1.0),
    32: (2.8, 1.0),
}


class MedicalRoomBase(Entry):
    def __init__(self, area, mic_type='mono'):
        self.area = area
        aspect_ratio = ROOM_ASPECT_RATIOS[self.area]
        width = math.sqrt(self.area / (aspect_ratio[0] / aspect_ratio[1]))
        length = width * (aspect_ratio[0] / aspect_ratio[1])
        self.dimensions = [length, width]
        self.mic_type = mic_type

    def create_room(self) -> tuple:
        length, width = self.dimensions
        height = 3.0
        room_dim = [length, width, height]

        print(f"Creating medical room ({length:.2f}m x {width:.2f}m x {height:.2f}m)...")
        room = Room.create_shoebox(room_dim, materials={
            "floor": get_material("linoleum"),
            "ceiling": get_material("ceiling_tile"),
            "walls": get_material("drywall")
        }, fs=DEFAULT_SAMPLING_RATE)

        mic = self._add_receiver(room)
        self._add_furniture(room)
        sources = self._add_sources(room)

        return room, sources, mic

    def _add_receiver(self, room: Room):
        raise NotImplementedError("Subclasses must implement this method")

    def _add_furniture(self, room: Room):
        raise NotImplementedError("Subclasses must implement this method")

    def _add_sources(self, room: Room) -> dict:
        raise NotImplementedError("Subclasses must implement this method")
