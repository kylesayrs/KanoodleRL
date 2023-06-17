from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from src.config import TrainingConfig, EnvironmentConfig
from src.piece import Piece
from src.environment import KanoodleEnvironment


def train_agent(training_config: TrainingConfig, environment_config: EnvironmentConfig):
    environment = make_vec_env(
        KanoodleEnvironment,
        env_kwargs={"environment_config": environment_config},
        n_envs=training_config.n_envs
    )

    model = PPO(
        training_config.policy,
        environment,
        policy_kwargs=training_config.policy_kwargs,
        learning_rate=training_config.learning_rate,
        n_steps=training_config.n_steps,
        batch_size=training_config.batch_size,
        n_epochs=training_config.n_epochs,
        gamma=training_config.gamma,
        gae_lambda=training_config.gae_lambda,
        clip_range=training_config.clip_range,
        verbose=training_config.verbosity,
        device=training_config.device,
    )

    model.learn(
        total_timesteps=training_config.total_timesteps,
        progress_bar=training_config.progress_bar
    )
    now_string = str(datetime.now()).replace(" ", "_")
    model.save(f"checkpoints/{now_string}.zip")


if __name__ == "__main__":
    training_config = TrainingConfig()
    environment_config = EnvironmentConfig(pieces_set_name="test")

    train_agent(training_config, environment_config)
