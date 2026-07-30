import torch
from torch.utils.data import DataLoader

from training.dataset import ChessDataset
from training.model import ChessPolicyNet

MODEL_PATH = "/kaggle/working/models/policy_net_epoch1.pth"
VAL_PATH = "/kaggle/input/datasets/satyajeet69/validation-set/val.csv"

BATCH_SIZE = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = ChessDataset(VAL_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
)

model = ChessPolicyNet().to(device)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()

correct1 = 0
correct5 = 0
total = 0

with torch.no_grad():

    for x, y in loader:

        x = x.to(device)
        y = y.to(device)

        outputs = model(x)

        # Top-1
        pred = outputs.argmax(dim=1)

        correct1 += (pred == y).sum().item()

        # Top-5
        top5 = outputs.topk(5, dim=1).indices

        correct5 += (
            top5 == y.unsqueeze(1)
        ).any(dim=1).sum().item()

        total += y.size(0)

print(f"Top-1 Accuracy : {100*correct1/total:.2f}%")
print(f"Top-5 Accuracy : {100*correct5/total:.2f}%")