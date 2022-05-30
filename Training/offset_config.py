from dataclasses import dataclass

from ExperimentRunners.OffsetRunner.Models import ReferencePoint

from vector import Vector2D
from color_generator import ColorGenerator

@dataclass
class OffsetConfig:
    drone_crop_size: int
    city_crop_size: int

    def generate_reference_points(self):
        reference_points = [
            ReferencePoint(Vector2D(5881, 1656), 'miškas'),
            ReferencePoint(Vector2D(15650, 24030), 'miškas'),
            ReferencePoint(Vector2D(8650, 2000), 'miškas'),
            ReferencePoint(Vector2D(5846, 4752), 'daugiabučiai'),
            ReferencePoint(Vector2D(8046, 19828), 'daugiabučiai'),
            ReferencePoint(Vector2D(3330, 9590), 'daugiabučiai'),
            ReferencePoint(Vector2D(5310, 7960), 'daugiabučiai'),
            ReferencePoint(Vector2D(2501, 3402), 'daugiabučiai'),
            ReferencePoint(Vector2D(1973, 22699), 'pieva'),
            ReferencePoint(Vector2D(4780, 4160), 'kelias'),
            ReferencePoint(Vector2D(14130, 1500), 'kelias'),
            ReferencePoint(Vector2D(24230, 10280), 'upė'),
            ReferencePoint(Vector2D(9960, 16340), 'gyvenvietė'),
            ReferencePoint(Vector2D(7781, 5645), 'miškas, kelias'),
            ReferencePoint(Vector2D(25000, 24570), 'miškas, kelias'),
            ReferencePoint(Vector2D(18543, 9141), 'miškas, daugiabučiai'),
            ReferencePoint(Vector2D(9250, 20580), 'daugiabučiai, kelias'),
            ReferencePoint(Vector2D(25678, 13083), 'daugiabučiai, gyvenvietė'),
            ReferencePoint(Vector2D(3298, 25260), 'kelias, pieva'),
            ReferencePoint(Vector2D(2013, 3788), 'kelias, gyvenvietė'),
        ]

        return reference_points

    def build_color_generator(self):
        label_colors = {
            'daugiabučiai': '#8e44ad',
            'kelias': '#34495e',
            'gyvenvietė': '#c0392b',
            'pieva': '#f1c40f',
            'miškas': '#16a085',
            'upė': '#3498db',
        }

        color_generator = ColorGenerator("#2c3e50", label_colors)
        
        return color_generator