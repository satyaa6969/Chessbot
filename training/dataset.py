import csv

import torch
from torch.utils.data import IterableDataset

from utils.encoder import fen_to_tensor
from utils.move_vocab import MOVE_TO_ID


class ChessDataset(IterableDataset):

    def __init__(self, csv_path):
        self.csv_path = csv_path

    def __iter__(self):

        with open(self.csv_path, "r", encoding="utf-8") as file:

            reader = csv.DictReader(file)

            for row in reader:

                x = fen_to_tensor(row["fen"])

                y = MOVE_TO_ID[row["move"]]

                yield x, torch.tensor(y, dtype=torch.long)