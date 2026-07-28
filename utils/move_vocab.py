import chess

MOVE_TO_ID = {}
ID_TO_MOVE = {}

move_id = 0

# Normal moves
for from_square in chess.SQUARES:
    for to_square in chess.SQUARES:
        move = chess.Move(from_square, to_square)

        uci = move.uci()

        MOVE_TO_ID[uci] = move_id
        ID_TO_MOVE[move_id] = uci
        move_id += 1

# Promotion moves
promotions = [
    chess.QUEEN,
    chess.ROOK,
    chess.BISHOP,
    chess.KNIGHT
]

for from_square in chess.SQUARES:

    rank = chess.square_rank(from_square)

    if rank not in [1, 6]:
        continue

    for to_square in chess.SQUARES:

        for promo in promotions:

            move = chess.Move(from_square, to_square, promotion=promo)

            uci = move.uci()

            if uci not in MOVE_TO_ID:
                MOVE_TO_ID[uci] = move_id
                ID_TO_MOVE[move_id] = uci
                move_id += 1

print(f"Vocabulary size: {len(MOVE_TO_ID)}")