from datetime import datetime

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.config import EnvironmentConfig, TrainingConfig, PPOConfig, DDPGConfig
from src.environment import KanoodleEnvironment


def train_agent(training_config: TrainingConfig, environment_config: EnvironmentConfig):
    environment = make_vec_env(
        KanoodleEnvironment,
        env_kwargs={"environment_config": environment_config},
        n_envs=training_config.n_envs
    )

    if isinstance(training_config, PPOConfig):
        model = PPO(
            training_config.policy,
            environment,
            policy_kwargs=training_config.policy_kwargs,
            learning_rate=training_config.learning_rate,
            learning_starts=training_config.learning_starts,
            n_steps=training_config.n_steps,
            batch_size=training_config.batch_size,
            n_epochs=training_config.n_epochs,
            gamma=training_config.gamma,
            gae_lambda=training_config.gae_lambda,
            clip_range=training_config.clip_range,
            normalize_advantage=training_config.normalize_advantage,
            verbose=training_config.verbose,
            device=training_config.device,
        )

    elif isinstance(training_config, DDPGConfig):
        model = DDPG(
            training_config.policy,
            environment,
            policy_kwargs=training_config.policy_kwargs,
            learning_rate=training_config.learning_rate,
            train_freq=training_config.train_freq,
            batch_size=training_config.batch_size,
            gamma=training_config.gamma,
            buffer_size=training_config.buffer_size,
            optimize_memory_usage=training_config.optimize_memory_usage,
            replay_buffer_class=training_config.replay_buffer_class,
            replay_buffer_kwargs=training_config.replay_buffer_kwargs,
            verbose=training_config.verbose,
            device=training_config.device,
        )

    model.learn(
        total_timesteps=training_config.total_timesteps,
        log_interval=training_config.log_interval,
        callback=EvalCallback(
            Monitor(KanoodleEnvironment(environment_config)),
            n_eval_episodes=training_config.n_eval_episodes,
            eval_freq=training_config.eval_freq,
            render=True,
        ) if training_config.n_eval_episodes > 0 else None,
        progress_bar=training_config.progress_bar,
    )
    now_string = str(datetime.now()).replace(" ", "_")
    model.save(f"checkpoints/{now_string}.zip")


if __name__ == "__main__":
    #training_config = PPOConfig()
    training_config = DDPGConfig()
    environment_config = EnvironmentConfig(
        board_shape=(5, 5),
        pieces_set_name="junior"
    )

    train_agent(training_config, environment_config)
