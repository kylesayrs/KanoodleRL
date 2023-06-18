import tqdm
import numpy

from src.config import EnvironmentConfig
from src.environment import KanoodleEnvironment


def validate_random_agent(environment_config: EnvironmentConfig, num_episodes: int, render: bool):
    environment = KanoodleEnvironment(environment_config)

    reward_returns = []
    successes = []
    for _ in tqdm.tqdm(range(num_episodes)):
        _observation = environment.reset()
        rewards = []
        is_finished = False
        if render:
            environment.render()
            
        while not is_finished:
            action_confs = numpy.ones(len(environment.actions))
            _observation, reward, is_finished, info = environment.step(action_confs)

            rewards.append(reward)
            if render:
                environment.render()

        successes.append(info["is_success"])
        reward_returns.append(sum(rewards))

    print(f"successes: {100 * numpy.mean(successes)}% +/- {numpy.std(successes):.2f}")
    print(f"returns  : {numpy.mean(reward_returns):.2f} +/- {numpy.std(reward_returns):.2f}")


if __name__ == "__main__":
    environment_config = EnvironmentConfig(
        board_shape=(3, 3),
        pieces_set_name="demo",
    )

    validate_random_agent(environment_config, num_episodes=10_000, render=False)
