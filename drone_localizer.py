from __future__ import annotations
from dataclasses import dataclass
import math
import time
import numpy as np
import random
from buslane.commands import Command, CommandBus

from WeightCalculators import WeightCalculator
from DroneProvider import DroneProvider
from clearable_command_bus import ClearableCommandBus
from map_provider import MapProvider
from timer import Timer
from vector import Vector2D

class StepResult():
    def __init__(self, drone_position: Vector2D, predicted_position: Vector2D, prediction_time: float):
        self.drone_position = drone_position
        self.predicted_position = predicted_position
        self.distance = drone_position.distance_to(predicted_position)
        self.prediction_time = prediction_time

    def dump(self):
        return {
            "drone_position": f"{self.drone_position}",
            "predicted_position": f"{self.predicted_position}",
            "distance": self.distance,
            "prediction_time": self.prediction_time,
        }

@dataclass
class StepCommand(Command):
    particles: np.ndarray
    weights: list[float]
    drone: DroneProvider
    probable_position: Vector2D
    cutting_time: float
    prediction_time: float

class DroneLocalizer():
    def __init__(self, particle_count: int, weight_calculator: WeightCalculator, particle_randomize_percent: float = 0):
        self.particle_count = particle_count
        self.particle_randomize_percent = particle_randomize_percent
        self.weight_calculator = weight_calculator

        self.command_bus = ClearableCommandBus()

    def _move_particles(self, city_map: MapProvider, particles, move_x_by, move_y_by, pixel_noise):
        """
        Moves particles in X and Y positions, off-setting movement with gaussian noise. 
        """
        moved_particles: list[tuple[int, int]] = []
        
        for x, y in particles:
            x_noise = int(random.gauss(0, pixel_noise))
            y_noise = int(random.gauss(0, pixel_noise))

            new_x = x + move_x_by + x_noise
            new_y = y + move_y_by + y_noise

            if city_map.is_in_map(new_x, new_y):
                moved_particles.append((new_x, new_y))
        
        if len(moved_particles) == 0:
            raise IndexError("All samples are outside of map, so drone is likely also out of map.")

        return np.array(moved_particles)

    def _get_highest_probability_particle(self, particles, weights: list) -> Vector2D:
        max_weight = max(weights)
        max_weight_index = weights.index(max_weight)

        probable_x, probable_y = particles[max_weight_index]

        return Vector2D(probable_x, probable_y)

    def _resample_particles(self, city_map: MapProvider, weights: list, particles: np.ndarray):
        """
        Take new particles according to weights. Higher weight, means higher probability for particle to be taken.
        Given particle can be taken any number of times.
        """

        # If no samples are under threshold, take random sample
        if np.sum(weights) == 0:
            resampled_particles = city_map.generate_random_locations(self.particle_count)
            print("\t\t Took all random particles")

            return resampled_particles

        randomize_count = math.floor(self.particle_count * self.particle_randomize_percent)
        random_particles = city_map.generate_random_locations(randomize_count)

        resample_count = self.particle_count - randomize_count
        resampled_indices = np.random.choice(len(particles), p=weights / np.sum(weights), size=resample_count)
        resampled_particles: np.ndarray = particles[resampled_indices]

        resampled_particles = np.concatenate((resampled_particles, random_particles), axis=0)

        return resampled_particles

    def fly_drone_route(self, drone: DroneProvider, city_map: MapProvider) -> list[StepResult]:
        # Create initial samples
        particles = city_map.generate_random_locations(self.particle_count)

        distances: list[StepResult] = []
        while drone.has_step():
            with Timer() as cutting_time:
                drone_image = drone.grab_image()
                drone_image_np = np.array(drone_image)

                particle_images = city_map.create_images_from_particles_threaded(particles)

            with Timer() as prediction_time:
                weights = self.weight_calculator.create_weights_from_images(drone_image_np, particle_images)
                probable_position = self._get_highest_probability_particle(particles, weights)

            distances.append(StepResult(drone.get_position(), probable_position, prediction_time.interval))

            # Data callback
            step_command = StepCommand(
                particles=particles,
                weights=weights,
                drone=drone,
                probable_position=probable_position,
                cutting_time=cutting_time.interval,
                prediction_time=prediction_time.interval
            )
            self.command_bus.execute(command=step_command)

            resampled_particles = self._resample_particles(city_map, weights, particles)

            # Move particles & Drone
            moved_by = drone.move_step()
            particles = self._move_particles(city_map, resampled_particles, moved_by.x, moved_by.y, 30)

        return distances