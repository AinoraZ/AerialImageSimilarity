import numpy as np
from dataclasses import dataclass

from . import ReferencePoint
from vector import Vector2D

@dataclass
class OffsetData:
    reference_point: ReferencePoint
    vector_offset: Vector2D
    original_weight: float
    weight: float
    offset_image: np.ndarray

    def __post_init__(self):
        self.delta = self.weight - self.original_weight
        self.real_distance =  self.reference_point.point.distance_to(self.reference_point.point + self.vector_offset)

    def dump(self):
        return {
            "offset": f"{self.vector_offset}",
            "real_distance": self.real_distance,
            "weight": self.weight,
            "delta": self.delta,
        }