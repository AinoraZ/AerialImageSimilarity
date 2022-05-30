import numpy as np
import statistics
from dataclasses import dataclass  

from . import ReferencePoint, DistanceData
from vector import Vector2D

@dataclass
class PointData:
    reference_point: ReferencePoint
    original_weight: float
    anchor_image: np.ndarray
    distances: list[DistanceData]

    def __post_init__(self):
        self.avg_delta = statistics.mean([distance.avg_delta for distance in self.distances])
        self.distances_dump = [distance.dump() for distance in self.distances]

    def dump(self):
        return {
            "label": self.reference_point.label,
            "reference_point": f"{self.reference_point.point}",
            "original_weight": self.original_weight,
            "avg_delta": self.avg_delta,
            "distances": self.distances_dump
        }