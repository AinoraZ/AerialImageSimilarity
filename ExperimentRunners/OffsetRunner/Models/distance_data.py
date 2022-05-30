from dataclasses import dataclass  
import statistics

from . import OffsetData

@dataclass
class DistanceData:
    target_distance: int
    offsets: list[OffsetData]

    def __post_init__(self):
        self.avg_delta = statistics.mean([offset.delta for offset in self.offsets])
        self.avg_weight = statistics.mean([offset.weight for offset in self.offsets])
        self.offsets_dump = [offset.dump() for offset in self.offsets]

    def dump(self):
        return {
            "target_distance": f"{self.target_distance}",
            "avg_weight": self.avg_weight,
            "avg_delta": self.avg_delta,
            "offsets": self.offsets_dump
        }