import chess
import torch

# Channel mapping
PIECE_TO_CHANNEL = {
    (chess.PAWN, chess.WHITE): 0,
    (chess.KNIGHT, chess.WHITE): 1,
    (chess.BISHOP, chess.WHITE): 2,
    (chess.ROOK, chess.WHITE): 3,
    (chess.QUEEN, chess.WHITE): 4,
    (chess.KING, chess.WHITE): 5,

    (chess.PAWN, chess.BLACK): 6,
    (chess.KNIGHT, chess.BLACK): 7,
    (chess.BISHOP, chess.BLACK): 8,
    (chess.ROOK, chess.BLACK): 9,
    (chess.QUEEN, chess.BLACK): 10,
    (chess.KING, chess.BLACK): 11,
}


def fen_to_tensor(fen: str) -> torch.Tensor:
    """
    Converts a FEN string into a tensor of shape [12, 8, 8]
    """

    board = chess.Board(fen)

    tensor = torch.zeros((12, 8, 8), dtype=torch.float32)

    for square, piece in board.piece_map().items():

        channel = PIECE_TO_CHANNEL[(piece.piece_type, piece.color)]

        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)

        tensor[channel, row, col] = 1.0

    return tensor