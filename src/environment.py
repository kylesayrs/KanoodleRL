from typing import List, Tuple

import numpy
from termcolor import colored
from gym import Env, spaces

from src.config import EnvironmentConfig
from src.piece import Piece
from pieces import load_pieces


class KanoodleEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig):
        super().__init__()
        self.config = environment_config

        self.pieces = load_pieces(environment_config.pieces_set_name)
        (
            self.actions,
            self.pieces_mask
        ) = get_actions(self.config.board_shape, self.pieces)

        self.observation_space = spaces.MultiDiscrete([
            numpy.prod(self.config.board_shape),
            len(self.pieces)
        ])
        self.action_space = spaces.Discrete(len(self.actions))

        self.reset()
        

    def reset(self):
        self.board = numpy.zeros(self.config.board_shape, dtype=int)
        self.available_pieces = list(range(len(self.pieces)))
        self.action_history = []


    def step(self, action_confs: int):
        if not self.invalid_actions_mask.all():
            # select action
            action_confs[self.invalid_actions_mask] = numpy.NINF
            action_index = numpy.argmax(action_confs)
            print(f"action_index: {action_index}")
            action = self.actions[action_index]
            piece_index = self.pieces_mask[action_index]

            # do action
            assert not numpy.bitwise_and(self.board, action).any()
            self.action_history.append([action, piece_index])
            self.board = numpy.bitwise_or(self.board, action)
            self.available_pieces.remove(piece_index)

        # return results
        observation = numpy.concatenate([self.board.flatten(), self.available_pieces])
        reward = self.get_reward()
        is_finished = self.is_finished()
        info = None

        return observation, reward, is_finished, info


    def get_reward(self):
        return (
            self.config.complete_reward
            if self.board.all()
            else self.config.step_reward
        )
    

    def is_finished(self):
        return self.board.all() or  self.invalid_actions_mask.all()


    @property
    def invalid_actions_mask(self):
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


    def render(self, mode="human"):
        if mode == "console":
            self._render_console()
        elif mode == "human":
            self._render_human()
        else:
            raise ValueError(f"Unknown render mode {mode}")


    def _render_console(self):
        print("_render_console")
        for y in range(self.config.board_shape[0]):
            for x in range(self.config.board_shape[1]):
                for action_mask, piece_index in self.action_history:
                    if action_mask[y, x]:
                        color = self.pieces[piece_index].color
                        print(colored("*", color=color), end=" ")
                        break
                else:
                    print(colored("o", color="white"), end=" ")
            print()

        print(f"{''.join(['- '] * self.config.board_shape[1])}")
        
        print(self.available_pieces)


    def _render_human(self):
        pass


def get_actions(board_shape: Tuple[int, int], pieces: List[Piece]):
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
