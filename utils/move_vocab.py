import chess

PROMOTION_RANKS = (1, 6)
PROMOTIONS = (
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT,
)


def build_move_vocab():
    move_to_id = {}
    id_to_move = {}

    move_id = 0

    # Normal moves
    for from_square in chess.SQUARES:
        for to_square in chess.SQUARES:
            uci = chess.Move(from_square, to_square).uci()

            move_to_id[uci] = move_id
            id_to_move[move_id] = uci
            move_id += 1

    # Promotion moves
    for from_square in chess.SQUARES:

        if chess.square_rank(from_square) not in PROMOTION_RANKS:
            continue

        for to_square in chess.SQUARES:
            for promotion in PROMOTIONS:

                uci = chess.Move(
                    from_square,
                    to_square,
                    promotion=promotion
                ).uci()

                if uci not in move_to_id:
                    move_to_id[uci] = move_id
                    id_to_move[move_id] = uci
                    move_id += 1

    return move_to_id, id_to_move


MOVE_TO_ID, ID_TO_MOVE = build_move_vocab()

VOCAB_SIZE = len(MOVE_TO_ID)


if __name__ == "__main__":
    print(f"Vocabulary size: {VOCAB_SIZE}")