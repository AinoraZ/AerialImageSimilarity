from vector import Vector2D

class ReferencePoint:
    def __init__(self, point: Vector2D, label: str = ""):
        self.point = point
        self.label = label

    def __str__(self):
        """Human-readable string representation of the vector."""

        return f'{self.label} - ({self.point.x}, {self.point.y})'

    def __repr__(self):
        """Unambiguous string representation of the vector."""

        return self.__str__()