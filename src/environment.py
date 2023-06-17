from typing import List, Tuple, Optional, Dict, Any

import numpy
from termcolor import colored
from gym import Env, spaces

from src.config import EnvironmentConfig
from src.piece import Piece
from src.utils import iterate_shape_2d
from pieces import load_pieces


class KanoodleEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig):
        super().__init__()
        self.config = environment_config

        self.pieces = load_pieces(environment_config.pieces_set_name)
        self.actions, self.pieces_mask = get_all_actions(
            self.config.board_shape, self.pieces
        )

        self.observation_space = spaces.Dict({
            "board": spaces.Box(0.0, 1.0, (numpy.prod(self.config.board_shape), )),
            "available_pieces_mask": spaces.Box(0.0, 1.0, (len(self.pieces), )),
        })
        self.action_space = spaces.Box(0.0, 1.0, (len(self.actions), ))

        self.reset()
        

    def reset(self) -> None:
        self.board = numpy.zeros(self.config.board_shape, dtype=int)
        self.available_pieces = list(range(len(self.pieces)))
        self.action_history = []

        return self.get_observation()


    def step(self, action_confs: int) -> Tuple[numpy.ndarray, float, bool, Dict[str, Any]]:
        if not self.invalid_actions_mask.all():
            # select action
            action_confs[self.invalid_actions_mask] = numpy.NINF
            action_index = numpy.argmax(action_confs)
            action = self.actions[action_index]
            piece_index = self.pieces_mask[action_index]

            # do action
            assert not numpy.bitwise_and(self.board, action).any()
            self.action_history.append((action, piece_index))
            self.board = numpy.bitwise_or(self.board, action)
            self.available_pieces.remove(piece_index)

        # return results
        observation = self.get_observation()
        reward = self.get_reward()
        is_finished = self.is_finished()
        info = {}

        return observation, reward, is_finished, info
    

    def get_observation(self) -> numpy.ndarray:
        available_pieces_mask = numpy.zeros(len(self.pieces), dtype=numpy.float32)
        available_pieces_mask[self.available_pieces] = 1.0
        return {
            "board": self.board.flatten(),
            "available_pieces_mask": available_pieces_mask
        }


    def get_reward(self) -> float:
        return (
            self.config.complete_reward
            if self.board.all()
            else self.config.step_reward
        )
    

    def is_finished(self) -> bool:
        return self.board.all() or self.invalid_actions_mask.all()


    @property
    def invalid_actions_mask(self) -> numpy.ndarray:
        unavailable_piece_actions = [
            peice_index not in self.available_pieces
            for peice_index in self.pieces_mask
        ]

        intersecting_piece_actions = [
            numpy.bitwise_and(self.board, action).any()
            for action in self.actions
        ]

        return numpy.bitwise_or(
            unavailable_piece_actions,
            intersecting_piece_actions
        )


    def render(self, mode: str = "console") -> None:
        if mode == "console":
            self.render_console()
        else:
            raise ValueError(f"Unknown render mode {mode}")


    def render_console(self, action_history: Optional[List[Tuple[numpy.ndarray, int]]] = None) -> None:
        action_history = self.action_history if action_history is None else action_history

        for y in range(self.config.board_shape[0]):
            for x in range(self.config.board_shape[1]):
                for action_mask, piece_index in action_history:
                    if action_mask[y, x]:
                        color = self.pieces[piece_index].color
                        print(colored(self.config.solid_char, color=color), end=" ")
                        break
                else:
                    print(colored(self.config.empty_char, color="white"), end=" ")
            print()
        
        print(self.available_pieces)

    def render_action(self, action: numpy.ndarray, piece_index: int):
        action_history = self.action_history.copy()
        action_history.append((action, piece_index))
        self.render_console(action_history)


def get_all_actions(board_shape: Tuple[int, int], pieces: List[Piece]) -> Tuple[numpy.ndarray, numpy.ndarray]:
    actions = []
    pieces_mask = []
    for piece_index, piece in enumerate(pieces):

        piece_actions = sum([
            iterate_shape_2d(board_shape, shape_variant)
            for shape_variant in piece.shape_variants
        ], [])

        peice_mask = [piece_index] * len(piece_actions)

        actions += piece_actions
        pieces_mask += peice_mask

    return actions, pieces_mask
