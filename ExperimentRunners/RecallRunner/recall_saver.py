from dataclasses import dataclass
import heapq
from random import sample
import numpy as np
from itertools import product
import math
import json
import os
from PIL import Image
from ExperimentRunners.RecallRunner.Models import RecallRunResult, RecommendationResult 
from map_provider import MapProvider
from vector import Vector2D

@dataclass
class RecallSaverOptions:
    city_map: MapProvider
    drone_map: MapProvider

    save_samples: tuple[int, int]

class RecallSaver:
    def __init__(self, base_folder: str, options: RecallSaverOptions):
        self.run_folder = f"{base_folder}/Recall"
        os.makedirs(self.run_folder, exist_ok=True)

        self.city_map = options.city_map
        self.drone_map = options.drone_map

        self.save_samples = options.save_samples

    def _build_point_folder(self, reference_point: Vector2D):
        point_dir = f"{self.run_folder}/{reference_point}"
        os.makedirs(point_dir, exist_ok=True)

        return point_dir

    def _save_stats(self, point_folder: str, relevant_distance: int, data: RecommendationResult):
        point_file = f"{point_folder}/recommendation-stats.json"

        with open(point_file, "w") as file:
            file.write(json.dumps(data.dump(relevant_distance), indent=4))

    def _save_image(self, point_folder, label, image_np, point: Vector2D):
        image = Image.fromarray(image_np)
        image.save(f"{point_folder}/{label}-{point}.png")

    def _save_sample_images(self, point_dir: str, reference_point: Vector2D, sample: RecommendationResult):
        anchor_image = self.drone_map.get_cropped_image(reference_point.x, reference_point.y)
        self._save_image(point_dir, "anchor", anchor_image, reference_point)

        for index, top_particle in enumerate(sample.top_particles):
            weight = f"{top_particle.weight:.4f}"

            top_image = self.city_map.get_cropped_image(top_particle.point.x, top_particle.point.y)
            self._save_image(point_dir, f"top{index}-w{weight}", top_image, top_particle.point)

        for index, bottom_particle in enumerate(sample.bottom_particles):
            weight = f"{bottom_particle.weight:.4f}"

            bottom_image = self.city_map.get_cropped_image(bottom_particle.point.x, bottom_particle.point.y)
            self._save_image(point_dir, f"bottom{index}-w{weight}", bottom_image, bottom_particle.point)

    def _get_distrubuted_samples(self, recall_result: RecallRunResult) -> list[RecommendationResult]:
        min_point, max_point = self.drone_map.get_boundaries()
        x_samples, y_samples = self.save_samples

        x_range = np.linspace(min_point.x, max_point.x, x_samples).astype(int).tolist()
        y_range = np.linspace(min_point.y, max_point.y, y_samples).astype(int).tolist()

        points = product(x_range, y_range)
        total_samples = np.prod(self.save_samples)
        sample_set = set()
        for x, y in points:
            target_point = Vector2D(x, y)
            closest_points = heapq.nsmallest(
                total_samples,
                recall_result.recommendations,
                key=lambda r: r.reference_point.distance_to(target_point))

            for closest in closest_points:
                if closest in sample_set:
                    continue
                
                # print(closest.reference_point, round(closest.reference_point.distance_to(target_point)))
                sample_set.add(closest)
                break

        return list(sample_set)

    def _save_distributed_samples(self, recall_result: RecallRunResult):
        random_samples = self._get_distrubuted_samples(recall_result)

        for sample in random_samples:
            reference_point = sample.reference_point
            point_dir = self._build_point_folder(reference_point)

            self._save_stats(point_dir, recall_result.relevant_distance, sample)
            self._save_sample_images(point_dir, reference_point, sample)
            
    def _save_run_stats(self, recall_result: RecallRunResult):
        run_file = f"{self.run_folder}/run-stats.json"

        with open(run_file, "w") as file:
            file.write(json.dumps(recall_result.dump(), indent=4))

    def save(self, recall_result: RecallRunResult):
        self._save_run_stats(recall_result)
        self._save_distributed_samples(recall_result)