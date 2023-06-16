from typing import List

import os
import numpy

from src.config import AgentConfig, EnvironmentConfig
from src.piece import Piece
from src.environment import KanoodleEnvironment
from pieces.standard_pieces import pieces


def create_action_space(pieces: List[Piece]):
    pass


def train_model(agent_config: AgentConfig, environment_config: EnvironmentConfig):
    print(agent_config)
    print(environment_config)
    print(environment_config.pieces)

    environment_config.pieces = []
    print(environment_config)

    

    #environment = KanoodleEnvironment(config.board_shape)
    pass
