from typing import Tuple

import numpy
import functools
import importlib

from gym import Env


class Immutable:
    _frozen = False


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = True


    def __delattr__(self, *args, **kwargs):
        if self._frozen:
            raise AttributeError("Cannot delete attribute from frozen object")
        object.__delattr__(self, *args, **kwargs)


    def __setattr__(self, *args, **kwargs):
        if self._frozen:
            raise AttributeError("Cannot set attribute of frozen object")
        object.__setattr__(self, *args, **kwargs)


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


def iterate_shape_2d(board_shape: Tuple[int, int], shape_variant: numpy.ndarray):
        masks = []

        for y in range(board_shape[0] - shape_variant.shape[0] + 1):
            for x in range(board_shape[1] - shape_variant.shape[1] + 1):
                board = numpy.zeros(board_shape, dtype=int)
                board[
                    y: y + shape_variant.shape[0],
                    x: x + shape_variant.shape[1]
                ] = shape_variant

                masks.append(board)
        
        return masks


def action_confs_to_prob(action_confs: numpy.ndarray, invalid_actions: numpy.ndarray):
    """
    TODO: reduce runtime a little
    """
    assert (action_confs >= 0).all()
    action_confs = action_confs.copy()

    action_confs[invalid_actions] = 0.0

    if sum(action_confs) == 0.0:
        action_confs[~invalid_actions] = 1.0

    return action_confs / sum(action_confs)


def get_fillable_spaces(actions, invalid_actions_mask) -> numpy.ndarray:
    return numpy.array(list(functools.reduce(
        numpy.bitwise_or,
        actions[~invalid_actions_mask],
    )), dtype=bool)


def loadModelConfig(training_config: "TrainingConfig", **kwargs):
    import src.config
    
    return getattr(src.config, f"{training_config.model_arch}Config")(**kwargs)

def loadModel(config: "ModelConfig", environment: Env, save_path: str = None):
    import stable_baselines3

    model_class_name = config.__class__.__name__.replace("Config", "")
    model_class = getattr(stable_baselines3, model_class_name)

    model_kwargs = config.dict()
    del model_kwargs["_frozen"]

    return model_class(
        env=environment,
        **model_kwargs
    )
