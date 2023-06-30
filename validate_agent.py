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
    unique_success_masks = []
    unique_success_histories = []
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
        if info["is_success"]:
            action_history_mask = environment.get_action_history_mask().tolist()
            if action_history_mask not in unique_success_masks:
                unique_success_masks.append(action_history_mask)
                unique_success_histories.append(environment.action_history)

    for action_history in unique_success_histories:
        environment.render_human(action_history, show_observation=False)
    
    print(f"unique successes: {len(unique_success_histories)}")
    print(f"success rate    : {100 * numpy.mean(successes)}% +/- {numpy.std(successes):.2f}")
    print(f"average return  : {numpy.mean(reward_returns):.2f} +/- {numpy.std(reward_returns):.2f}")


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
        render=False,
    )
