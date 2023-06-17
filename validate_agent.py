import sys

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.config import TrainingConfig, EnvironmentConfig
from src.piece import Piece
from src.environment import KanoodleEnvironment


def validate_agent(
    checkpoint_path: str,
    training_config: TrainingConfig,
    environment_config: EnvironmentConfig
):
    environment = KanoodleEnvironment(environment_config)

    model = PPO.load(checkpoint_path)

    observation = environment.reset()
    for i in range(training_config.num_validation_steps):
        action, _states = model.predict(observation)
        observation, rewards, dones, info = environment.step(action)
        environment.render()
        if dones:
            break


if __name__ == "__main__":
    checkpoint_path = sys.argv[1]

    training_config = TrainingConfig()
    environment_config = EnvironmentConfig(
        #board_shape=(3, 3),
        pieces_set_name="test",
    )

    validate_agent(
        checkpoint_path,
        training_config,
        environment_config
    )
