from __future__ import annotations
import math
import numpy as np
from IPython.display import display, Markdown
from vector import Vector2D

def chunk_list(source_list, n):
    elements_per_chunk = math.floor(len(source_list) / n)
    spare_elements = len(source_list) % n

    previous_end = 0
    split_lists = []
    for i in range(n):
        start_range = previous_end

        extra_item = 1 if i < spare_elements else 0
        end_range = start_range + elements_per_chunk + extra_item

        split_lists.append(source_list[start_range:end_range])
        previous_end = end_range

    return split_lists

def iterate_chunks(arr, chunk_size):
    iterator = iter(arr)
    current_chunk = []

    value = next(iterator, None)
    while value is not None:
        current_chunk.append(value)
        if len(current_chunk) >= chunk_size:
            yield current_chunk
            current_chunk = []

        value = next(iterator, None)

    if len(current_chunk) is not 0:
        yield current_chunk

def clamp(number : int, minimum: int, maximum: int):
    return np.clip(number, minimum, maximum)

def hex_color_dump(color: str, name: str = None):
    if name is None:
        name = color

    display(Markdown(f'<span style="font-family: monospace">{name} <span style="color: {color}">████</span></span>'))

def random_vector_of_length(radius):
    """
    Returns randomly pointing vector on a radius of a circle
    """
    random_vector = (np.random.rand(2) * 2) - 1
    pointing_vector = random_vector / np.linalg.norm(random_vector)

    pointing_vector2D = Vector2D(pointing_vector[0], pointing_vector[1])
    random_dist_vector2D = pointing_vector2D * radius
    
    int_vector = Vector2D(int(random_dist_vector2D.x), int(random_dist_vector2D.y))

    return int_vector

