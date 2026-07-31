import csv
from pathlib import Path

import torch
from torch.utils.data import IterableDataset, get_worker_info

from utils.encoder import fen_to_tensor
from utils.move_vocab import MOVE_TO_ID


class ChessDataset(IterableDataset):

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"Dataset not found:\n{self.csv_path}"
            )

    def __iter__(self):

        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        with self.csv_path.open("r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row_idx, row in enumerate(reader):

                if row_idx % num_workers != worker_id:
                    continue

                x = fen_to_tensor(row["fen"])
                y = MOVE_TO_ID[row["move"]]

                yield x, torch.tensor(y, dtype=torch.long)