import sys
import tqdm
import numpy

from stable_baselines3 import PPO, DDPG

from src.config import TrainingConfig, ModelConfig, EnvironmentConfig
from src.environment import KanoodleEnvironment
from src.utils import load_model_config, load_model


def validate_agent(
    checkpoint_path: str,
    model_config: ModelConfig,
    environment_config: EnvironmentConfig,
    greedy_policy: bool = False,
    num_episodes: int = 100,
    render: bool = True,
):
    environment = KanoodleEnvironment(environment_config)
    model = load_model(model_config, environment, checkpoint_path)

    reward_returns = []
    successes = []
    for _ in tqdm.tqdm(range(num_episodes)):
        observation = environment.reset()
        rewards = []
        is_finished = False

        while not is_finished:
            action_confs, _states = model.predict(observation)
            observation, reward, is_finished, info = environment.step(action_confs)

            rewards.append(reward)
            
        if render:
            environment.render()
        
        successes.append(info["is_success"])
        reward_returns.append(sum(rewards))

        # TODO: show all successes

    print(f"successes: {100 * numpy.mean(successes)}% +/- {numpy.std(successes):.2f}")
    print(f"returns  : {numpy.mean(reward_returns):.2f} +/- {numpy.std(reward_returns):.2f}")
    

if __name__ == "__main__":
    checkpoint_path = sys.argv[1]

    training_config = TrainingConfig()
    model_config = load_model_config(training_config)
    environment_config = EnvironmentConfig()

    validate_agent(
        checkpoint_path,
        model_config,
        environment_config,
        num_episodes=1_000,
        render=True,
    )
