from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "positions.csv"
TRAIN_CSV = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_CSV = PROJECT_ROOT / "data" / "processed" / "val.csv"

import pandas as pd



print("Loading CSV...")
df = pd.read_csv(INPUT_CSV)

print("Total positions:", len(df))

# Split by game_id, NOT by position
games = df["game_id"].unique()

split = int(0.95 * len(games))

train_games = set(games[:split])
val_games = set(games[split:])

train_df = df[df["game_id"].isin(train_games)]
val_df = df[df["game_id"].isin(val_games)]

print("Train positions:", len(train_df))
print("Validation positions:", len(val_df))

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)

print("Done.")