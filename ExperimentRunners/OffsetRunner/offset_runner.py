from __future__ import annotations
from dataclasses import dataclass  

import math
from map_provider import MapProvider
from tqdm.auto import tqdm
from vector import Vector2D
from WeightCalculators import ModelBasedWeightCalculator
from ModelBuilders import BaseModelBuilder
from WeightCalculators.Transformers import BaseTransformer

from .Models import ReferencePoint, OffsetData, DistanceData, PointData

@dataclass
class OffsetRunnerOptions:
    model_builder: BaseModelBuilder
    city_map: MapProvider
    drone_map: MapProvider

    reference_points: list[ReferencePoint]

class OffsetRunner:
    def __init__(self, options: OffsetRunnerOptions):
        self.target_distances = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 75, 100, 150, 250, 350, 500, 700, 1000]

        self.city_map = options.city_map
        self.drone_map = options.drone_map

        self.model_calculator = ModelBasedWeightCalculator(
            options.model_builder,
            batch_size=64, 
            transformer=BaseTransformer())

        self.reference_points = options.reference_points

    def _generate_offset_points(self, offset: int):
        vector_offsets = [
            Vector2D(offset, 0),
            Vector2D(0, offset),
            Vector2D(-offset, 0),
            Vector2D(0, -offset)
        ]

        def round_up(number):
            floored = math.floor(number)
            return floored if number - floored < 0.5 else floored + 1

        offset = round_up(math.sqrt((offset ** 2) / 2))
        side_offsets = [
            Vector2D(offset, offset),
            Vector2D(offset, -offset),
            Vector2D(-offset, offset),
            Vector2D(-offset, -offset)
        ]

        vector_offsets.extend(side_offsets)

        return vector_offsets

    def run(self) -> list[PointData]:
        print("Running offset runner tests")

        point_dump: list[PointData] = []
        for reference_point in tqdm(self.reference_points, desc="Running point offsets"):
            reference_point: ReferencePoint = reference_point

            anchor =  self.drone_map.get_cropped_image(reference_point.point.x, reference_point.point.y)
            positive = self.city_map.get_cropped_image(reference_point.point.x, reference_point.point.y)
            weights = self.model_calculator.create_weights_from_images(anchor, [positive])
            original_weight = weights[0]

            distances_dump = []
            for distance in self.target_distances:
                vector_offsets = self._generate_offset_points(distance)

                offset_points = [reference_point.point + vector_offset for vector_offset in vector_offsets]
                offset_particles = [(offset_point.x, offset_point.y) for offset_point in offset_points]

                offset_images = self.city_map.create_images_from_particles_threaded(offset_particles, 2)
                weights = self.model_calculator.create_weights_from_images(anchor, offset_images)

                offsets_dump = []
                for (weight, vector_offset, offset_image) in zip(weights, vector_offsets, offset_images):
                    offset_data = OffsetData(
                        reference_point = reference_point,
                        vector_offset = vector_offset,
                        original_weight = original_weight,
                        weight = weight,
                        offset_image = offset_image
                    )
                    
                    offsets_dump.append(offset_data)

                distance_data = DistanceData(target_distance=distance, offsets=offsets_dump)
                distances_dump.append(distance_data)

            point_data = PointData(
                reference_point=reference_point,
                original_weight=original_weight,
                anchor_image=anchor,
                distances=distances_dump)
            
            point_dump.append(point_data)

        print()

        return point_dump