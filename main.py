from torch.utils.data import DataLoader

from training.dataset import ChessDataset
from training.model import ChessPolicyNet

dataset = ChessDataset("data/processed/positions.csv")

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=0
)

model = ChessPolicyNet()

x, y = next(iter(loader))

print(x.shape)

out = model(x)

print(out.shape)