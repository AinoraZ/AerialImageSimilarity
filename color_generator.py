import hashlib
import math
from PIL import ImageColor
from dataclasses import dataclass  

@dataclass
class LabeledColor:
    label: str
    color: str

class ColorGenerator:
    def __init__(self, base_color, label_colors = None, color_count = 5000):
        self.base_color = base_color
        self.color_count = color_count
        self.label_colors = label_colors

    def _generate_hash(self, value: str) -> int:
        encoded = value.encode("utf-8")
        hash = hashlib.sha256(encoded)

        return int(hash.hexdigest(), 16) % self.color_count

    def _num_to_hex(self, value: int, padding: int) -> str:
        hexed = hex(value)[2:]
        hexed = hexed.zfill(padding)

        return f"{hexed}"

    def avg_color(self, colors: list[str]):
        tr = 0
        tg = 0
        tb = 0
        for color in colors:
            (r, g, b) = ImageColor.getrgb(color)
            
            tr += r
            tg += g
            tb += b

        nr = math.ceil((tr) / len(colors))
        ng = math.ceil((tg) / len(colors))
        nb = math.ceil((tb) / len(colors))

        hexR = self._num_to_hex(nr, 2)
        hexG = self._num_to_hex(ng, 2)
        hexB = self._num_to_hex(nb, 2)

        return f"#{hexR}{hexG}{hexB}"

    def random_color(self, value: str) -> str:
        hash = self._generate_hash(value)

        max_rgb = (255 * 255 * 255)

        random_rgb = math.floor(max_rgb * (hash / self.color_count))
        random_color = f"#{self._num_to_hex(random_rgb, 6)}"

        average_hex = self.avg_color([random_color, self.base_color])

        return average_hex

    def label_to_color(self, label: str) -> str:
        parts = [part.strip() for part in label.split(",")]

        colors = []
        for part in parts:
            if part in self.label_colors:
                colors.append(self.label_colors[part])
            else:
                colors.append(self.random_color(part))

        return self.avg_color(colors)