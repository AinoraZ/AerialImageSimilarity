from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
import numpy as np
from image_provider import ImageProvider
import numpy as np
import cv2

from vector import Vector2D
from util import chunk_list

class ImageProjection:
    def __init__(self, position: Vector2D, size: Vector2D):
        """
        Position - left bottom corner of new projected picture.
        Size - new size for picture.
        """
        self.position = position
        self.size = size

    def get_center_position(self) -> Vector2D:
        return self.position + (self.size / 2)

    def get_size(self) -> Vector2D:
        return self.size

    def __str__(self) -> str:
        return f"pos{self.position}-size{self.size}"

    def __repr__(self) -> str:
        return self.__str__()

class MapProvider():
    def __init__(self, image_provider: ImageProvider, crop_size: int, projection: ImageProjection = None):
        self.image_provider = image_provider
        self.projection = projection

        image = image_provider.get_image()
        width, height = image.size
        self.image_size = Vector2D(width, height)
 
        if projection is not None:
            image = self._project(image, projection)
            width, height = image.size
            self.image_size = Vector2D(width, height)

        self.image_np = np.asarray(image)
        self.crop_size = Vector2D(crop_size, crop_size)

        min_allowed_x = self.crop_size.x / 2
        min_allowed_y = self.crop_size.y / 2
        self.min_allowed = Vector2D(min_allowed_x, min_allowed_y)

        max_allowed_x = self.image_size.x - min_allowed_x
        max_allowed_y = self.image_size.y - min_allowed_y
        self.max_allowed =  Vector2D(max_allowed_x, max_allowed_y)


    def provider_label(self):
        parts = [self.image_provider.path(), f"cs{self.crop_size.x}"]

        if self.projection is not None:
            parts.append(f"{self.projection}")

        return "-".join(parts)

    def get_boundaries(self):
        return self.min_allowed, self.max_allowed

    def _get_crop_coords(self, crop_x, crop_y, crop_size: Vector2D):
        """
        Returns coordinates of image to crop converted from math coordinate system to pillow coordinate system.
        Math coords - (0, 0) bottom left.
        Pillow coords - (0, 0) is in the upper left.
        """
        left = int(crop_x - (crop_size.x / 2))
        right = int(left + crop_size.x)
        top = int((self.image_size.y - crop_y) - (crop_size.y / 2))
        bottom = int(top + crop_size.y)
        
        return (left, top, right, bottom)

    def _project(self, image: Image.Image, projection: ImageProjection) -> Image.Image:
        center = projection.get_center_position()
        (left, top, right, bottom) = self._get_crop_coords(center.x, center.y, projection.get_size())

        width, height = image.size
        if left < 0 or top < 0 or right > width or bottom > height:
            print(f"WARNING: crop was outside of map. Ignoring projection. {(left, top, right, bottom)}")
            return image

        return image.crop((left, top, right, bottom))

    def is_in_map(self, particle_x: int, particle_y: int):
        min_allowed, max_allowed = self.get_boundaries()

        return (min_allowed.x <= particle_x <= max_allowed.x) and (min_allowed.y <= particle_y <= max_allowed.y)

    def get_cropped_image(self, x, y):
        if not self.is_in_map(x, y):
            raise Exception(f"Trying to take particle outside of map... x: {x} y: {y} size: {self.crop_size} image: {self.image_size}.")

        (left, top, right, bottom) = self._get_crop_coords(x, y, self.crop_size)
        
        cropped_image = self.image_np[top:bottom, left:right]
        resized_image = cv2.resize(cropped_image, (224, 224), interpolation = cv2.INTER_AREA)

        return resized_image

    def _create_images_from_particles(self, particles: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
        images = {}
        for particle in particles:
            particle_x, particle_y = particle

            particle_image = self.get_cropped_image(particle_x, particle_y)
            images[(particle_x, particle_y)] = particle_image

        return images

    def create_images_from_particles_threaded(self, particles, nr_of_threads = 14) -> list[np.ndarray]:
        futures_list = []
        joined_images = {}

        # Remove duplicates
        unduped_particles = np.unique(particles, axis=0)
        particle_chunks = chunk_list(unduped_particles, nr_of_threads)

        with ThreadPoolExecutor(max_workers=nr_of_threads) as executor:
            for particle_chunk in particle_chunks:
                futures = executor.submit(self._create_images_from_particles, particle_chunk)
                futures_list.append(futures)

            for future in futures_list:
                result_images = future.result()
                joined_images = joined_images | result_images
        
        result_list = []
        for particle in particles:
            particle_x, particle_y = particle

            particle_image = joined_images[(particle_x, particle_y)]
            result_list.append(particle_image)

        return result_list

    def generate_random_location(self) -> tuple[int, int]:
        min_allowed, max_allowed = self.get_boundaries()
        return (np.random.randint(min_allowed.x, max_allowed.x), np.random.randint(min_allowed.y, max_allowed.y))

    def generate_random_locations(self, location_count):
        locations = [self.generate_random_location() for _ in range(location_count)]
        return np.array(locations)