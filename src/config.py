from typing import Tuple, List
from pydantic import BaseModel, Field


from src.piece import Piece
from src.utils import Immutable
from pieces import standard_pieces


class AgentConfig(Immutable, BaseModel):
    pass



class EnvironmentConfig(Immutable, BaseModel):
    board_shape: Tuple[int, int] = Field(default=(5, 11))
    pieces: List[Piece] = Field(default=standard_pieces)

    class Config:
        arbitrary_types_allowed = True
