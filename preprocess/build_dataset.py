from pathlib import Path
import csv
import chess.pgn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PGN_FILE = PROJECT_ROOT / "data" / "raw" / "elite_games.pgn"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "positions.csv"

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

if not PGN_FILE.exists():
    raise FileNotFoundError(f"PGN file not found:\n{PGN_FILE}")

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

            if ply < 8:
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