import chess.pgn
import csv
from tqdm import tqdm

PGN_FILE = "../data/raw/elite_games.pgn"
OUTPUT_FILE = "../data/processed/positions.csv"

with open(PGN_FILE, "r", encoding="utf-8") as pgn, \
     open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as csvfile:

    writer = csv.writer(csvfile)

    writer.writerow([
        "game_id",
        "position_id",
        "fen",
        "move",
        "white_elo",
        "black_elo",
        "result",
        "ply"
    ])

    game_id = 0
    position_id = 0

    while True:
        game = chess.pgn.read_game(pgn)

        if game is None:
            break

        board = game.board()

        white_elo = game.headers.get("WhiteElo", "")
        black_elo = game.headers.get("BlackElo", "")
        result = game.headers.get("Result", "")

        for ply, move in enumerate(game.mainline_moves()):

            if ply < 8:          # Skip opening theory
                board.push(move)
                continue

            writer.writerow([
                game_id,
                position_id,
                board.fen(),
                move.uci(),
                white_elo,
                black_elo,
                result,
                ply
            ])

            position_id += 1
            board.push(move)

        game_id += 1

print(f"Processed {game_id:,} games.")
print(f"Stored {position_id:,} positions.")