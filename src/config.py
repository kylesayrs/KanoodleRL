from typing import Tuple
from pydantic import BaseModel, Field

from src.piece import Piece
from src.utils import Immutable


class TrainingConfig(Immutable, BaseModel):
    n_envs: int = Field(default=2)
    total_timesteps: float = Field(default=1_000_000)

    policy: str = Field(default="MultiInputPolicy")
    policy_kwargs: str = Field(default={})

    learning_rate: float = Field(default=3e-6)
    n_steps: float = Field(default=1024, description="The number of steps to run for each environment per update")
    batch_size: int = Field(default=64)
    n_epochs: int = Field(default=15)

    gamma: float = Field(default=0.97)
    gae_lambda: float = Field(default=0.95)
    clip_range: float = Field(default=0.2)

    num_validation_steps: int = Field(default=300)
    progress_bar: bool = Field(default=False)
    verbosity: int = Field(default=2)
    device: str = Field(default="cpu")



class EnvironmentConfig(BaseModel):
    board_shape: Tuple[int, int] = Field(default=(5, 11))
    pieces_set_name: str = Field(default="standard")

    complete_reward: float = Field(default=100.0)
    fill_reward: float = Field(default=0.0)

    solid_char: str = Field(default="*")
    empty_char: str = Field(default="o")
