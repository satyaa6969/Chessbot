import chess.pgn

PGN_FILE = "../data/raw/elite_games.pgn"

with open(PGN_FILE, "r", encoding="utf-8") as pgn:
    game = chess.pgn.read_game(pgn)

    if game is None:
        print("No games found!")
        exit()

    print(f"White : {game.headers['White']}")
    print(f"Black : {game.headers['Black']}")
    print(f"Result: {game.headers['Result']}")

    moves = list(game.mainline_moves())
    print(f"Total moves: {len(moves)}")

    board = game.board()

    examples = []

    for move in game.mainline_moves():
        examples.append({
            "fen": board.fen(),
            "move": move.uci(),
            "turn": "white" if board.turn else "black"
        })
        board.push(move)

    print(f"Extracted {len(examples)} training examples.\n")

    for example in examples[:5]:
        print(example)