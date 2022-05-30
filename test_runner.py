from __future__ import annotations
from dataclasses import dataclass
import statistics
from DroneProvider import DroneProvider
import gc
from buslane.commands import Command
from clearable_command_bus import ClearableCommandBus

from drone_localizer import DroneLocalizer, StepResult
from map_provider import MapProvider
from timer import Timer


@dataclass
class RepeatResult(Command):
    drone_repr: str
    repeat_index: int
    step_results: list[StepResult]
    accuracy: float
    route_time: float

    def dump(self):
        return {
            "step_results": [result.dump() for result in self.step_results],
            "accuracy": self.accuracy,
            "route_time": self.route_time,
            "average_prediction_time": statistics.mean([result.prediction_time for result in self.step_results]),
            "average_distance": statistics.mean([result.distance for result in self.step_results]),
            "distance_std": statistics.stdev([result.distance for result in self.step_results]),
        }

class DroneResult():
    def __init__(self, drone_repr: str, repeat_results: list[RepeatResult]):
        self.drone_repr = drone_repr
        self.repeat_results = repeat_results
        self.average_accuracy = statistics.mean([repeat.accuracy for repeat in repeat_results])

    def get_distances(self):
        distances = []
        for repeat in self.repeat_results:
            for step in repeat.step_results:
                distances.append(step.distance)

        return distances

    def avg_distances(self):
        return statistics.mean(self.get_distances())

    def distances_std(self):
        return statistics.stdev(self.get_distances())

    def dump(self):
        return {
            "drone": self.drone_repr,
            "average_accuracy": self.average_accuracy,
            "repeat_results": [repeat.dump() for repeat in self.repeat_results],
            "total_time": sum([repeat.route_time for repeat in self.repeat_results]),
            "average_distance": self.avg_distances(),
            "distance_std": self.distances_std(),
        }

@dataclass
class TestRun:
    name: str
    drone: DroneProvider
    city_map: MapProvider

class TestRunner:
    def __init__(self, localizer: DroneLocalizer, accuracy_threshold: float, repeat_count: int = 10):
        self.localizer = localizer
        self.accuracy_threshold = accuracy_threshold
        self.repeat_count = repeat_count

        # Events
        self.command_bus = ClearableCommandBus()

    # Increasing accuracy
    def _mean_accuracy(self, step_results: list[StepResult]):
        accuracy = 0
        for step_result in step_results:
            if step_result.distance <= self.accuracy_threshold:
                accuracy += 1

        return accuracy / len(step_results)

    def run(self, test_run: TestRun) -> DroneResult:
        drone, city_map = test_run.drone, test_run.city_map

        repeat_results: list[RepeatResult] = []
        for repeat in range(self.repeat_count):
            print(f"\t Starting Repeat {repeat}.")
            try:
                with Timer() as route_time:
                    step_results = self.localizer.fly_drone_route(drone, city_map)
            except IndexError as error:
                print(f"Failed with exception: {error}. Skipping.")

                drone.reset_route()
                gc.collect()
                continue

            repeat_result = RepeatResult(
                drone_repr=f"{drone}",
                repeat_index=repeat,
                step_results=step_results,
                accuracy=self._mean_accuracy(step_results),
                route_time=route_time.interval)

            repeat_results.append(repeat_result)
            print(f"\t Ended Repeat {repeat}. Accuracy: {repeat_result.accuracy}. Elapsed time: {route_time.interval}")

            self.command_bus.execute(command=repeat_result)
            drone.reset_route()
            
            # Workaround for tensorflow memory leak
            gc.collect()

        return DroneResult(
            drone_repr=f"{drone}",
            repeat_results=repeat_results)
