from dataclasses import dataclass
from itertools import product
import numpy as np
import gc
import math
import heapq

from tqdm.auto import tqdm
from map_provider import MapProvider
from ModelBuilders import BaseModelBuilder
from WeightCalculators import ModelBasedWeightCalculator
from WeightCalculators.Transformers import BaseTransformer

from util import iterate_chunks
from vector import Vector2D
from .Models import ParticleEmbedding, WeightedParticleEmbedding, RecommendationResult, RecallRunResult

@dataclass
class RecallRunnerOptions:
    model_builder: BaseModelBuilder
    city_map: MapProvider
    drone_map: MapProvider

    step_size: int
    top: int
    relevant_distance: int
    drone_locations: list[tuple[int, int]]

class RecalRunner:
    def __init__(self, options: RecallRunnerOptions):
        self.chunk_size = 1000
        self.step_size = options.step_size

        self.top = options.top
        self.relevant_distance = options.relevant_distance

        self.model = options.model_builder.create_model(True)
        self.drone_locations = options.drone_locations

        self.city_map = options.city_map
        self.drone_map = options.drone_map

        self.model_calculator = ModelBasedWeightCalculator(
            options.model_builder,
            batch_size=64, 
            transformer=BaseTransformer())

    def _load_recommendation_db(self):
        min_point, max_point = self.city_map.get_boundaries()

        x_range = range(int(min_point.x), int(max_point.x), self.step_size)
        y_range = range(int(min_point.y), int(max_point.y), self.step_size)
        total = len(x_range) * len(y_range)

        particles = product(x_range, y_range)

        particle_embs: list[ParticleEmbedding] = []

        total_chunks = math.ceil(total / self.chunk_size)
        for chunk in tqdm(iterate_chunks(particles, self.chunk_size), total=total_chunks, desc="Creating recommendation db"):
            images = self.city_map.create_images_from_particles_threaded(chunk)
            results = self.model.predict(np.array(images), batch_size=64)
            images = []

            chunked_embs = [
                ParticleEmbedding(
                    point=Vector2D(particle[0], particle[1]), 
                    embedding=result) 
                for particle, result in zip(chunk, results)
            ]

            particle_embs.extend(chunked_embs)

            gc.collect()

        return particle_embs

    def _calculate_recommendations(self, location_embeddings, recommendation_db: list[ParticleEmbedding]) -> list[RecommendationResult]:
        recommendation_embs = [recommendation.embedding for recommendation in recommendation_db]
        
        recommendation_results: list[RecommendationResult] = []
        for (drone_x, drone_y), drone_embeding in tqdm(location_embeddings, desc="Processing recall locations"):
            drone_position = Vector2D(drone_x, drone_y)

            weights = self.model_calculator.calculate_weights(drone_embeding, recommendation_embs)

            top_recommendations = heapq.nlargest(self.top, zip(recommendation_db, weights), key=lambda r: r[1])
            bottom_recommendations = heapq.nsmallest(self.top, zip(recommendation_db, weights), key=lambda r: r[1])

            top_particles = [WeightedParticleEmbedding(top, weight) for top, weight in top_recommendations]
            bottom_particles = [WeightedParticleEmbedding(bottom, weight) for bottom, weight in bottom_recommendations]

            recommendation_result = RecommendationResult(
                reference_point=drone_position,
                top_particles=top_particles,
                bottom_particles=bottom_particles)

            recommendation_results.append(recommendation_result)

        print()
        return recommendation_results

    def run(self) -> RecallRunResult:
        recommendation_db = self._load_recommendation_db()

        drone_images = self.drone_map.create_images_from_particles_threaded(self.drone_locations)
        drone_embeddings = self.model.predict(np.array(drone_images), batch_size=64)
        drone_images = []

        drone_location_embeddings = list(zip(self.drone_locations, drone_embeddings))

        recommendation_results = self._calculate_recommendations(
            location_embeddings=drone_location_embeddings,
            recommendation_db=recommendation_db)
            
        return RecallRunResult(
            city_provider_label=self.city_map.provider_label(),
            drone_provider_label=self.drone_map.provider_label(),

            step_size=self.step_size,
            top=self.top,
            relevant_distance=self.relevant_distance,
            recommendations=recommendation_results)