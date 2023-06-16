from typing import Tuple
from pydantic import BaseModel, Field

import termcolor

from src.piece import Piece
from src.utils import Immutable


class AgentConfig(Immutable, BaseModel):
    pass



class EnvironmentConfig(BaseModel):
    board_shape: Tuple[int, int] = Field(default=(5, 11))
    pieces_set_name: str = Field(default="standard")

    complete_reward: float = Field(default=1.0)
    step_reward: float = Field(default=0.0)

    solid_char: str = Field(default="*")
    empty_char: str = Field(default="o")
