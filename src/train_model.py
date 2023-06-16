from typing import List

import os
import numpy

from src.config import AgentConfig, EnvironmentConfig
from src.piece import Piece
from src.environment import KanoodleEnvironment


def create_action_space(pieces: List[Piece]):
    pass


def train_model(agent_config: AgentConfig, environment_config: EnvironmentConfig):
    print(agent_config)
    print(environment_config)

    environment = KanoodleEnvironment(environment_config)

    action_confs = numpy.array(range(len(environment.actions)), dtype=numpy.float32)

    is_finished = False
    while not is_finished:
        observation, reward, is_finished, info = environment.step(action_confs)
        environment.render(mode="console")
