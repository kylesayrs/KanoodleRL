from typing import List, Tuple, Optional, Dict, Any, Union

import numpy
import functools
from termcolor import colored
from gym import Env, spaces

from src.config import EnvironmentConfig
from src.piece import Piece
from src.utils import iterate_shape_2d, action_confs_to_prob, get_fillable_spaces, rand_argmax
from pieces import load_pieces


class KanoodleEnvironment(Env):
    def __init__(self, environment_config: EnvironmentConfig):
        super().__init__()
        self.validate_config(environment_config)
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

        # spaces
        self.observation_space = self.make_observation_space()
        self.action_space = self.make_action_space()

        self.reset()


    def validate_config(self, config):
        pass


    def make_observation_space(self):
        all_spaces_dict = {
            "board": spaces.Box(0.0, 1.0, (numpy.prod(self.config.board_shape), )),
            "board_image": spaces.Box(0.0, 1.0, self.config.board_shape),
            "action_history_mask": spaces.Box(0.0, 1.0, (len(self.actions), )),
            "available_pieces_mask": spaces.Box(0.0, 1.0, (len(self.pieces), )),
            "invalid_actions_mask": spaces.Box(0.0, 1.0, (len(self.actions), )),
        }

        spaces_dict = {
            space_name: space
            for space_name, space in all_spaces_dict.items()
            if space_name in self.config.observation_spaces
        }

        return spaces.Dict(spaces_dict)


    def make_action_space(self):
        if self.config.discrete_actions:
            return spaces.Discrete(len(self.actions))
        else:
            return spaces.Box(0.0, 100, (len(self.actions), ))


    def reset(self) -> None:
        self.board = numpy.zeros(self.config.board_shape, dtype=int)
        self.invalid_actions_mask = self._initial_invalid_actions.copy()
        self.action_history = []

        return self.get_observation()
    

    def get_action(self, model_output):
        if self.config.discrete_actions:
            action_index = model_output
            if self.config.prevent_invalid_actions and self.invalid_actions_mask[action_index]:
                action_index = numpy.random.choice(
                    range(len(self.actions)),
                    p=action_confs_to_prob(numpy.ones((len(self.actions), )), self.invalid_actions_mask)
                )
        else:
            if self.config.prevent_invalid_actions:
                model_output[self.invalid_actions_mask] = numpy.NINF
            action_index = rand_argmax(model_output)
            #action_index = numpy.argmax(model_output)

        return (
            action_index,
            self.actions[action_index],
            self.pieces_mask[action_index]
        )


    def step(self, model_output: Union[int, numpy.ndarray]) -> Tuple[numpy.ndarray, float, bool, Dict[str, Any]]:
        action_index, action, piece_index = self.get_action(model_output)

        # punish invalid action
        if self.invalid_actions_mask[action_index]:
            self.invalid_actions_mask.fill(True)

        # do valid action
        else:
            assert not (self.board & action).any()
            self.board |= action
            self.invalid_actions_mask = self.update_invalid_actions(action_index, piece_index)
            self.action_history.append(action_index)

        # return results
        observation = self.get_observation()
        reward = self.get_reward()
        is_finished = self.is_finished()
        info = {"is_success": self.is_success()}

        return observation, reward, is_finished, info
    

    def get_observation(self) -> numpy.ndarray:
        observation_dict = {}

        if "board" in self.config.observation_spaces:
            observation_dict["board"] = self.board.flatten()

        if "board_image" in self.config.observation_spaces:
            observation_dict["board_image"] = self.board

        if "action_history_mask" in self.config.observation_spaces:
            observation_dict["action_history_mask"] = self.get_action_history_mask()

        if "available_pieces_mask" in self.config.observation_spaces:
            available_pieces_mask = numpy.zeros(len(self.pieces), dtype=numpy.float32)
            available_pieces_mask[self.available_pieces] = 1.0
            observation_dict["available_pieces_mask"] = available_pieces_mask

        if "invalid_actions_mask" in self.config.observation_spaces:
            observation_dict["invalid_actions_mask"] = self.invalid_actions_mask


        return observation_dict
    

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
        if self.config.calc_unsolvable:
            self.invalid_actions_mask |= self.get_unsolvable_actions_mask()

        return self.invalid_actions_mask


    def render(self, mode: str = "human") -> None:
        if mode == "human":
            self.render_human()
        else:
            raise ValueError(f"Unknown render mode {mode}")


    def render_human(
        self,
        action_history: Optional[List[int]] = None,
        show_observation: bool = True
    ) -> None:
        action_history = self.action_history if action_history is None else action_history

        for y in range(self.config.board_shape[0]):
            for x in range(self.config.board_shape[1]):
                for action_index in action_history:
                    action = self.actions[action_index]
                    piece_index = self.pieces_mask[action_index]
                    if action[y, x]:
                        color = self.pieces[piece_index].color
                        print(colored(self.config.solid_char, color=color), end=" ")
                        break
                else:
                    print(colored(self.config.empty_char, color="white"), end=" ")
            print()
        
        if show_observation:
            observation = self.get_observation()
            #print(observation["action_history_mask"])
            #print(observation["available_pieces_mask"])
            print(f"reward: {self.get_reward()}")
            print(f"is_finished: {self.is_finished()}")

        print("-----------------")


    @property
    def available_pieces(self):
        return numpy.unique(self.pieces_mask[~self.invalid_actions_mask])


    def render_action(self, action_index: int):
        action_history = self.action_history.copy()
        action_history.append(action_index)
        self.render_human(action_history)


    def get_action_history_mask(self):
        action_history_mask = numpy.zeros((len(self.actions)), dtype=int)
        action_history_mask[self.action_history] = 1

        return action_history_mask


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
