import statistics

from . import DistanceStats, SampleTriplet

class TrainingStats:
    def __init__(self, sample_triplets: list[SampleTriplet], positive_distances: list[float], negative_distances: list[float], time: float):
        self.sample_triplets = sample_triplets
        self.positive_distances = positive_distances
        self.negative_distances = negative_distances
        self.time = time

    def positive_stats(self) -> DistanceStats:
        return DistanceStats(self.positive_distances)

    def negative_stats(self) -> DistanceStats:
        return DistanceStats(self.negative_distances)

    def accuracy(self):
        accurate = 0
        total = 0

        for positive, negative in zip(self.positive_distances, self.negative_distances):
            if positive < negative:
                accurate += 1
            
            total += 1

        return round(accurate / total, 2)

    def avg_diff(self):
        differences = []
        for positive, negative in zip(self.positive_distances, self.negative_distances):
            differences.append(negative - positive)

        return statistics.mean(differences)

    def dump(self):
        return {
            "accuracy": self.accuracy(),
            "time": self.time,
            "average_difference": self.avg_diff(),
            "positive_stats": self.positive_stats().dump(),
            "negative_stats": self.negative_stats().dump(),
            "distances": [(positive, negative) for positive, negative in zip(self.positive_distances, self.negative_distances)],
        }