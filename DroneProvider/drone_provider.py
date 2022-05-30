from __future__ import annotations
from vector import Vector2D
from PIL import Image


class DroneProvider:
    def __init__(self):
        pass

    def has_step(self) -> bool:
        pass

    def get_current_step(self) -> int:
        pass

    def move_step(self) -> Vector2D:
        pass

    def grab_image(self) -> Image.Image:
        pass

    def get_position(self) -> Vector2D:
        pass

    def get_reset_count(self) -> int:
        pass

    def reset_route(self):
        pass