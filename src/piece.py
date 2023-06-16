from typing import List, Any

import numpy

from src.utils import variants_equal, get_unique_variants


class Piece():
    base_shape: numpy.ndarray
    shape_variants: List[numpy.ndarray]

    def __init__(self, base_shape: List[List[int]], color: str="white"):
        self.base_shape = numpy.array(base_shape)
        self.color = color

        self.shape_variants = self._generate_shape_variants(base_shape)


    def _generate_shape_variants(self, base_shape: List[List[int]]):
        base_shape = numpy.array(base_shape)
        
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
            (self.base_shape == other.base_shape).all()
        )
    


