import numpy

from src.config import EnvironmentConfig
from src.environment import KanoodleEnvironment


def get_piece_input(environment: KanoodleEnvironment, reward: float):
    print(chr(27) + "[2J")
    for piece_index in environment.available_pieces:
        print(f"({piece_index}): ")
        environment.pieces[piece_index].print()
    print(f"{''.join(['- '] * environment_config.board_shape[1])}")
    environment.render()

    action_piece_index = None
    while action_piece_index == None:
        try:
            action_piece_index = int(input(f"Which piece?: "))
        except EOFError:
            exit(0)
        except Exception as exception:
            print(exception)

    return action_piece_index


def get_action_index_input(environment: KanoodleEnvironment, action_piece_index: int):
    print(chr(27) + "[2J")
    for action_index, action in enumerate(environment.actions):
        if (
            environment.pieces_mask[action_index] == action_piece_index and
            not environment.invalid_actions_mask[action_index]
        ):
            print(f"[{action_index}]: ")
            environment.render_action(action_index)

    action_index = None
    while action_index == None:
        try:
            action_index = int(input(f"Which location?: "))
        except EOFError:
            exit(0)
        except Exception as exception:
            print(exception)

    return action_index


def play_game(environment_config: EnvironmentConfig):
    environment = KanoodleEnvironment(environment_config)

    total_rewards = 0.0
    reward = None
    is_finished = False
    while not is_finished:
        action_piece_index = get_piece_input(environment, reward)

        action_index = get_action_index_input(environment, action_piece_index)

        if environment_config.discrete_actions:
            player_output = action_index
        else:
            player_output = numpy.zeros(len(environment.actions))
            player_output[action_index] = 1.0

        observation, reward, is_finished, info = environment.step(player_output)
        total_rewards += reward

    environment.render()


if __name__ == "__main__":
    environment_config = EnvironmentConfig()

    play_game(environment_config)
