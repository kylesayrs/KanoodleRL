from typing import List, Tuple, Optional
from pydantic import BaseModel, Field

import torch

from src.utils import Immutable


class TrainingConfig(Immutable, BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=100_000)  # demo: 15k, junior: 100k
    log_interval: int = Field(default=10)

    model_arch: str = "DQN"

    wandb_mode: str = Field(default="disabled")
    n_eval_episodes: int = Field(default=0)
    eval_freq: int = Field(default=1_000, description="num steps")
    eval_render: bool = Field(default=False)

    progress_bar: bool = Field(default=False)


class ModelConfig(Immutable, BaseModel):
    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={
        #"activation_fn": torch.nn.ReLU,
        #"net_arch": [512, 512, 512]
        #"net_arch": [512, 256, 128, 64, 32]
        #"net_arch": [128, 128, 128]
    })

    verbose: int = Field(default=2)
    device: str = Field(default="cpu")
    tensorboard_log: str = Field(default="./tensorboard")

    class Config:
        arbitrary_types_allowed = True


class DQNConfig(ModelConfig):
    learning_starts: int = Field(default=128)
    learning_rate: float = Field(default=1e-3)  # demo: 1e-2, junior: 1e-3
    train_freq: Tuple[int, str] = Field(default=(10, "step"))
    batch_size: int = Field(default=128)
    gamma: float = Field(default=0.9)

    exploration_fraction: float = Field(default=0.1)
    exploration_initial_eps: float = Field(default=1.0)
    exploration_final_eps: float = Field(default=0.05)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)


class DDPGConfig(ModelConfig):
    learning_starts: int = Field(default=128)
    learning_rate: float = Field(default=3e-6)  # demo: 1e-5, junior: 1e-6
    train_freq: Tuple[int, str] = Field(default=(10, "step"))
    batch_size: int = Field(default=128)
    gamma: float = Field(default=0.9)

    action_noise: Optional[str] = Field(default="Normal")
    action_noise_mu: float = Field(default=0.0)
    action_noise_sigma: float = Field(default=0.03)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)


class PPOConfig(ModelConfig):
    learning_rate: float = Field(default=3e-7)
    n_steps: float = Field(default=10, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=128)
    n_epochs: int = Field(default=5)

    ent_coef: float = Field(default=0.1)

    gamma: float = Field(default=0.99)
    gae_lambda: float = Field(default=0.99)
    clip_range: float = Field(default=0.2)
    normalize_advantage: bool = Field(default=True)


class EnvironmentConfig(BaseModel):
    board_shape: Tuple[int, int] = Field(default=(11, 5))
    pieces_set_name: str = Field(default="standard")
    num_starting_pieces: int = Field(default=0)

    discrete_actions: bool = Field(default=True)
    prevent_invalid_actions: bool = Field(default=True)
    calc_unsolvable: bool = Field(default=True)

    observation_spaces: List[str] = [
        #"board",
        #"board_image",
        "action_history_mask",
        #"available_pieces_mask",
        #"invalid_actions_mask"    
    ]

    complete_reward: float = Field(default=10.0)
    fail_reward: float = Field(default=-10.0)
    fill_reward: float = Field(default=0.0)
    step_reward: float = Field(default=0.0)

    solid_char: str = Field(default="*")
    empty_char: str = Field(default="o")
