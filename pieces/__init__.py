from typing import List

import termcolor

from src.piece import Piece


def load_pieces(pieces_set_name: str) -> List[Piece]:
    def pieces_are_unique(pieces: List[Piece]):
        for piece_i_index, piece_i in enumerate(pieces):
            for piece_j in pieces[(piece_i_index + 1):]:
                assert piece_i != piece_j
                
        return True
    

    def piece_colors(pieces: List[Piece]):
        for piece in pieces:
            assert piece.color in termcolor.COLORS

        return True
    

    if pieces_set_name == "standard":
        from .standard_pieces import pieces
    
    elif pieces_set_name == "test":
        from .test_pieces import pieces
    
    else:
        raise ValueError(f"Unknown piece set name {pieces_set_name}")

    assert pieces_are_unique(pieces)
    assert piece_colors(pieces)

    return pieces
    
