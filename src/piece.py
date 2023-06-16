from typing import List, Any, Tuple
import pydantic

import numpy
from termcolor import colored

from src.utils import variants_equal, get_unique_variants


class Piece():
    base_shape: numpy.ndarray
    shape_variants: List[numpy.ndarray]

    def __init__(self, base_shape: List[List[int]], color: str="white"):
        self.base_shape = numpy.array(base_shape, dtype=int)
        self.color = color

        self.shape_variants = self._generate_shape_variants(base_shape)


    def _generate_shape_variants(self, base_shape: List[List[int]]):
        base_shape = numpy.array(base_shape, dtype=int)
        
        variants = []

        variants += [
            numpy.rot90(base_shape, k)
            for k in range(4)
        ]

        if not variants_equal(numpy.flip(base_shape, axis=0), base_shape):
            vertical_flipped = numpy.flip(base_shape, axis=0)
            variants += [
                numpy.rot90(vertical_flipped, k)
                for k in range(4)
            ]

        if not variants_equal(numpy.flip(base_shape, axis=1), base_shape):
            horizontal_flipped = numpy.flip(base_shape, axis=1)
            variants += [
                numpy.rot90(horizontal_flipped, k)
                for k in range(4)
            ]

        unique_variants = get_unique_variants(variants)

        return unique_variants
    

    def __eq__(self, other: Any):
        return (
            isinstance(other, Piece) and
            (
                variants_equal(self.base_shape, other.base_shape) or
                self.color == other.color
            )
        )
    

    def print(self):
        for y in range(self.base_shape.shape[0]):
            for x in range(self.base_shape.shape[1]):
                if self.base_shape[y, x]:
                    print(colored("*", color=self.color), end=" ")
                else:
                    print(colored(" ", color="white"), end=" ")
            print()


def piece_validator(value: Any) -> Piece:
    if isinstance(value, Piece):
        return value
    if value == "Piece":
        return Piece()
    raise ValueError("Must be a Piece or the string 'Piece'")


pydantic.validators._VALIDATORS.append((Piece, [piece_validator]))
