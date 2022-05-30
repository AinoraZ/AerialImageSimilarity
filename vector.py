from __future__ import annotations
from math import sqrt, atan2, pow
from typing import Union

FloatInt = Union[float, int]

class Vector2D:
    def __init__(self, x: FloatInt, y: FloatInt):
        self.x, self.y = x, y

    def __repr__(self):
        return repr((self.x, self.y))

    def __str__(self):
        return f'{self.__repr__()}'

    def __sub__(self, other: 'Vector2D'):
        return Vector2D(self.x - other.x, self.y - other.y)

    def __add__(self, other: 'Vector2D'):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __mul__(self, number: Union[float, int]):
        if isinstance(number, int) or isinstance(number, float):
            return Vector2D(self.x * number, self.y * number)

        raise NotImplementedError()

    def dot(self, other: 'Vector2D'):
        if not isinstance(other, Vector2D):
            raise TypeError()

        return (self.x * other.x) + (self.y * other.y)

    __matmul__ = dot

    def __rmul__(self, number: FloatInt):
        return self.__mul__(number)

    def __neg__(self):
        return Vector2D(-self.x, -self.y)

    def __truediv__(self, number: FloatInt):
        return Vector2D(self.x / number, self.y / number)

    def __mod__(self, number: FloatInt):
        return Vector2D(self.x % number, self.y % number)

    def __abs__(self):
        return sqrt(pow(self.x, 2) + pow(self.y, 2))

    def distance_to(self, other: 'Vector2D'):
        return abs(self - other)

    def to_polar(self):
        return self.__abs__(), atan2(self.y, self.x)