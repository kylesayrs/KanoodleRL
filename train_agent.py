
from datetime import datetime

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

import wandb
from wandb.integration.sb3 import WandbCallback

from src.config import EnvironmentConfig, TrainingConfig, ModelConfig
from src.environment import KanoodleEnvironment
from src.utils import load_model, load_model_config


def init_wandb(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    environment_config: EnvironmentConfig
):
    wandb.init(
        project="KanoodleRL",
        mode="online",
        sync_tensorboard=True,
        config={
            "training_config": training_config.dict(),
            "model_config": model_config.dict(),
            "environment_config": environment_config.dict(),
        }
    )


def makeEvalCallback(training_config: TrainingConfig):
    return EvalCallback(
        Monitor(KanoodleEnvironment(environment_config)),
        n_eval_episodes=training_config.n_eval_episodes,
        eval_freq=training_config.eval_freq,
        render=training_config.eval_render,
    )


def makeCallbacks(training_config: TrainingConfig, model_config: ModelConfig):
    wandb_callback = WandbCallback(
        verbose=model_config.verbose,
    )

    if training_config.n_eval_episodes > 0:
        return [
            wandb_callback,
            makeEvalCallback(
                training_config
            )
        ]

    return wandb_callback


def train_agent(
    training_config: TrainingConfig,
    model_config: ModelConfig,
    environment_config: EnvironmentConfig
):
    init_wandb(training_config, model_config, environment_config)

    environment = make_vec_env(
        KanoodleEnvironment,
        env_kwargs={"environment_config": environment_config},
        n_envs=training_config.n_envs
    )

    model = load_model(model_config, environment)

    model.learn(
        total_timesteps=training_config.total_timesteps,
        log_interval=training_config.log_interval,
        callback=makeCallbacks(training_config, model_config),
        progress_bar=training_config.progress_bar,
    )
    now_string = str(datetime.now()).replace(" ", "_")
    save_path = f"checkpoints/{now_string}.zip"
    model.save(save_path)
    print(f"saved model to {save_path}")


if __name__ == "__main__":
    training_config = TrainingConfig()
    model_config = load_model_config(training_config)
    environment_config = EnvironmentConfig()

    train_agent(training_config, model_config, environment_config)
