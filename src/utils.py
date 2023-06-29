from typing import Tuple

import numpy
import functools
import importlib

import stable_baselines3
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
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


def load_model_config(training_config: "TrainingConfig", **kwargs):
    import src.config
    
    return getattr(src.config, f"{training_config.model_arch}Config")(**kwargs)


def load_model(config: "ModelConfig", environment: DummyVecEnv, save_path: str = None):
    model_class_name = config.__class__.__name__.replace("Config", "")
    model_class = getattr(stable_baselines3, model_class_name)

    if not save_path:
        model_kwargs = config.dict()
        del model_kwargs["_frozen"]

        if "action_noise" in model_kwargs:
            model_kwargs["action_noise"] = make_action_noise(model_kwargs, environment)
            del model_kwargs["action_noise_mu"]
            del model_kwargs["action_noise_sigma"]

        return model_class(
            env=environment,
            **model_kwargs,
        )
    
    else:
        return model_class.load(
            save_path,
            env=environment,
        )


def make_action_noise(model_kwargs, environment):
    if model_kwargs["action_noise"] is None:
        return None

    action_noise_class_name = f"{model_kwargs['action_noise']}ActionNoise"
    import stable_baselines3.common.noise as SB3Noise
    action_noise_class = getattr(SB3Noise, action_noise_class_name)
    
    env_actions = environment.get_attr("actions")[0]

    action_space = numpy.full((len(env_actions), ), 1.0)
    mu = action_space * model_kwargs["action_noise_mu"]
    sigma = action_space * model_kwargs["action_noise_sigma"]
    
    return action_noise_class(mu, sigma)


def rand_argmax(array):
    return numpy.random.choice(
        numpy.where(array == array.max())[0]
    )
