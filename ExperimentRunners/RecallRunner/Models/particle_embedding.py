from vector import Vector2D
import numpy as np

class ParticleEmbedding:
    def __init__(self, point: Vector2D, embedding: np.ndarray):
        self.point = point
        self.embedding = embedding

class WeightedParticleEmbedding:
    def __init__(self, particle: ParticleEmbedding, weight: float):
        self.point = particle.point
        self.embedding = particle.embedding
        self.weight = weight

    def dump(self, reference_point: Vector2D):
        return {
            "point": f"{self.point}",
            "distance_from_reference": f"{reference_point.distance_to(self.point)}",
            "weight": self.weight,
        }