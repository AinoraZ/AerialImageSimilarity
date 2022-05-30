from dataclasses import dataclass

from . import WeightedParticleEmbedding
from vector import Vector2D

@dataclass
class RecommendationResult:
    reference_point: Vector2D
    top_particles: list[WeightedParticleEmbedding]
    bottom_particles: list[WeightedParticleEmbedding]

    def __init__(self, reference_point: Vector2D, top_particles: list[WeightedParticleEmbedding], bottom_particles: list[WeightedParticleEmbedding]):
        self.reference_point = reference_point
        self.top_particles = top_particles
        self.bottom_particles = bottom_particles

    def __eq__(self, other: 'RecommendationResult') -> bool:
        return self.reference_point.x == other.reference_point.x and self.reference_point.y == other.reference_point.y

    def __hash__(self):
        return hash(f"{self.reference_point}")

    def good_recommendations(self, relevant_distance):
        count = 0
        for top_particle in self.top_particles:
            if top_particle.point.distance_to(self.reference_point) <= relevant_distance:
                count += 1

        return count

    def total_recommendations(self):
        return len(self.top_particles)

    def dump(self, relevant_distance: int):
        return {
            "reference_point": f"{self.reference_point}",
            "good_recommendations": self.good_recommendations(relevant_distance),
            "total_recommendations": self.total_recommendations(),
            "top_particles": [particle.dump(self.reference_point) for particle in self.top_particles],
            "bottom_particles": [particle.dump(self.reference_point) for particle in self.bottom_particles],
        }