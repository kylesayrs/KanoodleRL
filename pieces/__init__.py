from typing import List

import termcolor

from src.piece import Piece


def load_pieces(pieces_set_name: str) -> List[Piece]:
    def piece_colors(pieces: List[Piece]):
        for piece in pieces:
            assert piece.color in termcolor.COLORS

        return True
    

    if pieces_set_name == "standard":
        from .standard_pieces import pieces

    elif pieces_set_name == "junior":
        from .junior_pieces import pieces

    elif pieces_set_name == "small":
        from .small_pieces import pieces
    
    else:
        raise ValueError(f"Unknown piece set name {pieces_set_name}")

    assert piece_colors(pieces)

    return pieces
    
