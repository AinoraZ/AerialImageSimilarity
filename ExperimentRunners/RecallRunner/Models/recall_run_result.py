from dataclasses import dataclass

from map_provider import ImageProjection
from . import RecommendationResult

@dataclass
class RecallRunResult:
    city_provider_label: str
    drone_provider_label: str

    step_size: int
    top: int
    relevant_distance: int
    recommendations: list[RecommendationResult]

    def dump(self):
        return {
            "city_provider_label": self.city_provider_label,
            "drone_provider_label": self.drone_provider_label,

            "step_size": self.step_size,
            "top": self.top,
            "relevant_distance": self.relevant_distance,
            "samples": len(self.recommendations),

            "precision": self.precision(),
            "recommendations": [recommendation.dump(self.relevant_distance) for recommendation in self.recommendations],
        }

    def precision(self):
        good = 0
        total = 0

        for recommendation in self.recommendations:
            good += recommendation.good_recommendations(self.relevant_distance)
            total += recommendation.total_recommendations()

        return good / total