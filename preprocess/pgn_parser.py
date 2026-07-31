from pathlib import Path
import chess.pgn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PGN_FILE = PROJECT_ROOT / "data" / "raw" / "elite_games.pgn"

with open(PGN_FILE, "r", encoding="utf-8") as pgn:
    game = chess.pgn.read_game(pgn)

    if game is None:
        raise SystemExit("No games found!")

    print(f"White : {game.headers['White']}")
    print(f"Black : {game.headers['Black']}")
    print(f"Result: {game.headers['Result']}")

    board = game.board()

    examples = []

    for move in game.mainline_moves():
        examples.append({
            "fen": board.fen(),
            "move": move.uci(),
            "turn": "white" if board.turn else "black"
        })
        board.push(move)

    print(f"Total moves: {len(examples)}")
    print(f"Extracted {len(examples)} training examples.\n")

    for example in examples[:5]:
        print(example)