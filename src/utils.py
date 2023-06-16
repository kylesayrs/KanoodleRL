import numpy


def variants_equal(variant_1: numpy.ndarray, variant_2: numpy.ndarray):
    return variant_1.shape == variant_2.shape and (variant_1 == variant_2).all()


def get_unique_variants(array: numpy.ndarray):
    unique_elements = []

    for element_i in array:
        for element_j in unique_elements:
            if variants_equal(element_i, element_j):
                break

        else:
            unique_elements.append(element_i)
            

    return unique_elements
