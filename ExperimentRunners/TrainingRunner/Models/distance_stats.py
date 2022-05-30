import statistics

class DistanceStats:
    def __init__(self, distances):
        self.distances = distances

    def min(self):
        return min(self.distances)

    def max(self):
        return max(self.distances)

    def avg(self):
        return statistics.mean(self.distances)

    def median(self):
        return statistics.median(self.distances)

    def dump(self):
        return {
            "min": self.min(),
            "max": self.max(),
            "avg": self.avg(),
            "median": self.median(),
        }