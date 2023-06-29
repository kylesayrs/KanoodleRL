from typing import Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

import torch
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import NormalActionNoise

from src.utils import Immutable


class TrainingConfig(Immutable, BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=1000_000)  # demo: 15k

    model_arch: str = "DDPG"

    log_interval: int = Field(default=1)
    n_eval_episodes: int = Field(default=0)
    eval_freq: int = Field(default=1_000, description="num steps")
    eval_render: bool = Field(default=False)

    progress_bar: bool = Field(default=False)


class ModelConfig(Immutable, BaseModel):
    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={
        "activation_fn": torch.nn.ReLU
    })

    verbose: int = Field(default=2)
    device: str = Field(default="cpu")
    tensorboard_log: str = Field(default="./tensorboard")

    class Config:
        arbitrary_types_allowed = True


class DDPGConfig(ModelConfig):
    learning_starts: int = Field(default=128)
    learning_rate: float = Field(default=1e-6)  # demo: 1e-5
    train_freq: Tuple[int, str] = Field(default=(10, "step"))
    batch_size: int = Field(default=128)
    gamma: float = Field(default=0.90)

    action_noise: Optional[str] = Field(default="Normal")
    action_noise_mu: float = Field(default=0.0)
    action_noise_sigma: float = Field(default=0.15)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)
    replay_buffer_class: Optional[ReplayBuffer] = Field(default=None)
    replay_buffer_kwargs: Optional[Dict[str, Any]] = Field(default=None)


class DQNConfig(ModelConfig):
    learning_starts: int = Field(default=128)
    learning_rate: float = Field(default=1e-4)  # demo: 1e-5
    train_freq: Tuple[int, str] = Field(default=(10, "step"))
    batch_size: int = Field(default=128)
    gamma: float = Field(default=0.90)

    exploration_fraction: float = Field(default=1.0)
    exploration_initial_eps: float = Field(1.0)
    exploration_final_eps: float = Field(0.05)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)
    replay_buffer_class: Optional[ReplayBuffer] = Field(default=None)
    replay_buffer_kwargs: Optional[Dict[str, Any]] = Field(default=None)


class PPOConfig(ModelConfig):
    learning_rate: float = Field(default=7e-6)
    n_steps: float = Field(default=32, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=8)
    n_epochs: int = Field(default=5)

    gamma: float = Field(default=0.99)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)
    normalize_advantage: bool = Field(default=True)


class EnvironmentConfig(BaseModel):
    board_shape: Tuple[int, int] = Field(default=(5, 5))
    pieces_set_name: str = Field(default="junior")

    discrete: bool = Field(default=False)
    prevent_invalid_actions: bool = Field(default=True)
    calc_unsolvable: bool = Field(default=True)

    complete_reward: float = Field(default=10.0)
    fail_reward: float = Field(default=-10.0)
    fill_reward: float = Field(default=0.0)
    step_reward: float = Field(default=0.0)

    solid_char: str = Field(default="*")
    empty_char: str = Field(default="o")
