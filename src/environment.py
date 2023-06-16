from typing import List, Tuple

import numpy
from gym import Env, spaces

from src.piece import Piece, create_action_masks


class KanoodleEnvironment(Env):
    def __init__(self, board_shape: Tuple[int, int], pieces: List[Piece]):
        super().__init__()

        self.observation_space = spaces.MultiDiscrete([numpy.prod(board_shape), len(pieces)])
        
        self.action_masks = create_action_masks(board_shape, pieces)
        self.action_space = spaces.Discrete(len(self.action_masks))

        self.reset()
        
    def reset():
        pass


    def step():
        pass


    def render(self, mode="human"):
        if mode == "console":
            self._render_console()
        elif mode == "human":
            self._render_human()
        else:
            raise ValueError(f"Unknown render mode {mode}")


    def _render_console(self):
        pass


    def _render_human(self):
        pass
