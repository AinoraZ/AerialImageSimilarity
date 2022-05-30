from __future__ import annotations

from PIL import Image

from DroneProvider import DroneProvider
from vector import Vector2D
from map_provider import MapProvider

class SimulatedDroneProvider(DroneProvider):
    def __init__(self, map_provider: MapProvider, start_position: Vector2D, move_by: Vector2D, step_count: int = 20):
        self.__start_position = start_position
        self.position = start_position
        self.move_by = move_by
        self.step_count = step_count
        self.map_provider = map_provider

        self.__current_step = 0
        self.__reset_count = 0

    def __repr__(self) -> str:
        return f"{repr(self.__start_position)}_{repr(self.move_by)}"

    def __str__(self) -> str:
        return self.__repr__()

    def get_position(self) -> Vector2D:
        return self.position

    def get_current_step(self) -> int:
        return self.__current_step

    def has_step(self) -> bool:
        return self.__current_step < self.step_count

    def move_step(self) -> None:
        if not self.has_step():
            return Vector2D(0, 0)

        self.position += self.move_by
        self.__current_step += 1

        return self.move_by

    def grab_image(self) -> Image.Image:
        return self.map_provider.get_cropped_image(self.position.x, self.position.y)

    def get_reset_count(self) -> int:
        return self.__reset_count

    def reset_route(self):
        self.position = self.__start_position
        self.__current_step = 0
        self.__reset_count += 1
