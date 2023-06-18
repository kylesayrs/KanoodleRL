import sys
import tqdm
import numpy

from stable_baselines3 import PPO, DDPG

from src.config import TrainingConfig, DDPGConfig, PPOConfig, EnvironmentConfig
from src.environment import KanoodleEnvironment


def validate_agent(
    checkpoint_path: str,
    training_config: TrainingConfig,
    environment_config: EnvironmentConfig,
    num_episodes: int = 100,
    render: bool = True,
):
    environment = KanoodleEnvironment(environment_config)

    if isinstance(training_config, PPOConfig):
        model = PPO.load(checkpoint_path)

    elif isinstance(training_config, DDPGConfig):
        model = DDPG.load(checkpoint_path)

    else:
        raise ValueError(f"Unknown training config class {training_config.__class__}")

    observation = environment.reset()
    reward_returns = []
    successes = []
    for _ in tqdm.tqdm(range(num_episodes)):
        rewards = []
        is_finished = False
        while not is_finished:
            action_confs, _states = model.predict(observation)
            print(action_confs)
            observation, reward, is_finished, info = environment.step(action_confs)

            rewards.append(reward)
            if render:
                environment.render()
        
        successes.append(info["is_success"])
        reward_returns.append(sum(rewards))
        environment.reset()

    print(f"successes: {100 * numpy.mean(successes):1f}% {sum(successes)}/{len(successes)}")
    print(f"returns  : {numpy.mean(reward_returns):.2f} +/- {numpy.std(reward_returns):.2f}")


if __name__ == "__main__":
    checkpoint_path = sys.argv[1]

    training_config = DDPGConfig()
    environment_config = EnvironmentConfig(
        board_shape=(3, 3),
        pieces_set_name="demo",
    )

    validate_agent(
        checkpoint_path,
        training_config,
        environment_config,
        num_episodes=1_000, render=True
    )
