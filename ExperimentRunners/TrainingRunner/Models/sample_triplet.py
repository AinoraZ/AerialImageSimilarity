from dataclasses import dataclass
import numpy as np

from . import ImageResult

@dataclass
class SampleTriplet:
    anchor: ImageResult
    positive: ImageResult
    negative: ImageResult