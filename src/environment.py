from typing import List, Tuple, Optional, Dict, Any

import numpy
import functools
from termcolor import colored
from gym import Env, spaces

from src.config import EnvironmentConfig
from src.piece import Piece
from src.utils import iterate_shape_2d, action_confs_to_prob, get_fillable_spaces
from pieces import load_pieces


class KanoodleEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig):
        super().__init__()
        self.config = environment_config

        # load game
        self.board = numpy.zeros(self.config.board_shape, dtype=int)
        self.pieces = load_pieces(environment_config.pieces_set_name)
        self.actions, self.pieces_mask = get_all_actions(
            self.config.board_shape, self.pieces
        )

        # precomputed values
        self._intersecting_actions_masks = get_intersecting_actions_masks(self.actions)
        self._initial_invalid_actions = self.update_invalid_actions()

        self.observation_space = spaces.Dict({
            "board": spaces.Box(0.0, 1.0, (numpy.prod(self.config.board_shape), )),
            #"board_image": spaces.Box(0.0, 1.0, self.config.board_shape),
            "available_pieces_mask": spaces.Box(0.0, 1.0, (len(self.pieces), )),
        })
        self.action_space = spaces.Box(0.0, 1.0, (len(self.actions), ))

        self.reset()
        

    def reset(self) -> None:
        self.board = numpy.zeros(self.config.board_shape, dtype=int)
        self.invalid_actions_mask = self._initial_invalid_actions.copy()
        self.action_history = []

        return self.get_observation()


    def step(self, action_confs: numpy.ndarray) -> Tuple[numpy.ndarray, float, bool, Dict[str, Any]]:
        #print(action_confs)
        action_prob = action_confs_to_prob(action_confs, self.invalid_actions_mask)
        #action_index = numpy.random.choice(list(range(len(action_prob))), p=action_prob)
        action_index = numpy.argmax(action_prob)
        action = self.actions[action_index]
        piece_index = self.pieces_mask[action_index]
        #print(action_prob)

        # do action
        assert not (self.board & action).any()
        self.board |= action
        self.invalid_actions_mask = self.update_invalid_actions(action_index, piece_index)
        self.action_history.append((action, piece_index))

        # return results
        observation = self.get_observation()
        reward = self.get_reward()
        is_finished = self.is_finished()
        info = {"is_success": self.is_success()}

        return observation, reward, is_finished, info
    

    def get_observation(self) -> numpy.ndarray:
        available_pieces_mask = numpy.zeros(len(self.pieces), dtype=numpy.float32)
        available_pieces_mask[self.available_pieces] = 1.0
        return {
            "board": self.board.flatten(),
            #"board_image": self.board,
            "available_pieces_mask": available_pieces_mask
        }
    

    def is_success(self):
        return self.board.all()
    

    def is_failure(self):
        return self.invalid_actions_mask.all()


    def get_reward(self) -> float:
        if self.is_success():
            return self.config.complete_reward
        
        elif self.is_failure():
            return self.config.fail_reward
        
        else:
            return (
                self.config.fill_reward * numpy.count_nonzero(self.board) +
                self.config.step_reward
            )
    

    def is_finished(self) -> bool:
        return self.is_success() or self.is_failure()
            

    def get_unsolvable_actions_mask(self):
        unsolvable_actions = []
        for action_index, (action, action_piece_index) in enumerate(zip(self.actions, self.pieces_mask)):
            if self.invalid_actions_mask[action_index]:
                unsolvable_actions.append(True)
                continue

            board_after_action = self.board | action
            if board_after_action.all():
                unsolvable_actions.append(False)
                continue

            invalid_actions_after_action = self.invalid_actions_mask.copy()
            invalid_actions_after_action |= self.pieces_mask == action_piece_index
            invalid_actions_after_action |= self._intersecting_actions_masks[action_index]
            if invalid_actions_after_action.all():
                unsolvable_actions.append(True)
                continue

            fillable_spaces_after_action = get_fillable_spaces(self.actions, invalid_actions_after_action)
            if not (board_after_action | fillable_spaces_after_action).all():
                unsolvable_actions.append(True)
                continue
            else:
                unsolvable_actions.append(False)
                continue

        assert len(unsolvable_actions) == len(self.actions)
        return numpy.array(unsolvable_actions)
    

    def update_invalid_actions(self, action_index: Optional[int] = None, chosen_piece_index: Optional[int] = None) -> numpy.ndarray:
        if action_index is None and chosen_piece_index is None:
            self.invalid_actions_mask = numpy.full((len(self.actions), ), False)
        
        else:
            # piece is not available
            self.invalid_actions_mask |= self.pieces_mask == chosen_piece_index

            # actions would intersect
            self.invalid_actions_mask |= self._intersecting_actions_masks[action_index]

        # actions that lead to an unsolvable board
        self.invalid_actions_mask |= self.get_unsolvable_actions_mask()

        return self.invalid_actions_mask


    def render(self, mode: str = "human") -> None:
        if mode == "human":
            self.render_human()
        else:
            raise ValueError(f"Unknown render mode {mode}")


    def render_human(self, action_history: Optional[List[Tuple[numpy.ndarray, int]]] = None) -> None:
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
        
        print(self.get_observation()["available_pieces_mask"])
        print(f"reward: {self.get_reward()}")
        print(f"is_finished: {self.is_finished()}")


    @property
    def available_pieces(self):
        return numpy.unique(self.pieces_mask[~self.invalid_actions_mask])


    def render_action(self, action: numpy.ndarray, piece_index: int):
        action_history = self.action_history.copy()
        action_history.append((action, piece_index))
        self.render_human(action_history)


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

    return numpy.array(actions), numpy.array(pieces_mask)


def get_intersecting_actions_masks(actions: List[numpy.ndarray]):
    return [
        [
            action2[action1 == 1].any()
            for action2 in actions
        ]
        for action1 in actions
    ]
