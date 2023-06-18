from typing import Tuple, Dict, Any, Optional
from pydantic import BaseModel, Field

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3 import HerReplayBuffer

from src.utils import Immutable


class TrainingConfig(Immutable, BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=500_000)

    log_interval: int = Field(default=100)
    n_eval_episodes: int = Field(default=0)
    eval_freq: int = Field(default=1_000, description="num steps")

    progress_bar: bool = Field(default=False)

    class Config:
        arbitrary_types_allowed = True


class DDPGConfig(TrainingConfig):
    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={})

    learning_starts: int = Field(default=0)
    learning_rate: float = Field(default=1e-5)
    train_freq: Tuple[int, str] = Field(default=(100, "step"))
    batch_size: int = Field(default=64)
    gamma: float = Field(default=0.99)

    buffer_size: int = Field(default=100_000)
    optimize_memory_usage: bool = Field(default=False)
    replay_buffer_class: Optional[ReplayBuffer] = Field(default=None)
    replay_buffer_kwargs: Optional[Dict[str, Any]] = Field(default=None)

    verbose: int = Field(default=2)
    device: str = Field(default="cpu")


class PPOConfig(TrainingConfig):
    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={})

    learning_rate: float = Field(default=3e-6)
    n_steps: float = Field(default=1024, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=64)
    n_epochs: int = Field(default=15)

    gamma: float = Field(default=0.99)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)
    normalize_advantage: bool = Field(default=True)

    verbose: int = Field(default=1)
    device: str = Field(default="cpu")


class EnvironmentConfig(BaseModel):
    board_shape: Tuple[int, int] = Field(default=(5, 11))
    pieces_set_name: str = Field(default="standard")

    complete_reward: float = Field(default=10.0)
    fail_reward: float = Field(default=-10.0)
    fill_reward: float = Field(default=0.0)
    step_reward: float = Field(default=1.0)

    solid_char: str = Field(default="*")
    empty_char: str = Field(default="o")
