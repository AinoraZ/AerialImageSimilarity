from __future__ import annotations
from PIL import Image
from vector import Vector2D

Image.MAX_IMAGE_PIXELS = 1355497489

class ImageProvider:
    def __init__(self, image_path: str):
        self.image_path = image_path

        print(f"Loading image: {image_path}")
        self.image = Image.open(image_path)
        self.image.load()

    def path(self):
        return self.image_path

    def get_image(self):
        return self.image
    
    def close(self):
      self.image.close()