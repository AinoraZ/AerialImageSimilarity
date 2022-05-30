import numpy as np
from dataclasses import dataclass

@dataclass
class ImageResult:
    image: np.ndarray
    embedding: np.ndarray