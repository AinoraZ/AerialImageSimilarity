import numpy as np

from . import BaseTransformer

class StretchedTanTransformer(BaseTransformer):
    def __init__(self, images_threshold):
        self.images_threshold = images_threshold

    def _tan(self, number):
        tanged = np.tan(number * (np.pi / 2)) * 3
        return tanged

    def _normalize_range(self, weight):
        if weight < self.images_threshold:
            return 0

        return (weight - self.images_threshold) / (1 - self.images_threshold)

    def transform(self, weight):
        weight = self._normalize_range(weight)
        weight = min(0.99999, weight) #Fix bug when distance is 1

        if weight == 0:
            return 0

        tan_weight = self._tan(weight)
        tan_weight = min(100000, tan_weight) #Fix bug when distance reaches infinity

        return tan_weight