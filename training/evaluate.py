import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.dataset import ChessDataset
from training.model import ChessPolicyNet

# ======================================
# Configuration
# ======================================

MODEL_PATH = "/kaggle/working/models/policy_net_epoch1.pth"
VAL_PATH = "/kaggle/input/datasets/satyajeet69/Validation-Set/val.csv"

BATCH_SIZE = 256

assert os.path.exists(MODEL_PATH), f"Model not found:\n{MODEL_PATH}"
assert os.path.exists(VAL_PATH), f"Validation set not found:\n{VAL_PATH}"

# ======================================
# Device
# ======================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ======================================
# Dataset
# ======================================

dataset = ChessDataset(VAL_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=True
)

# ======================================
# Model
# ======================================

model = ChessPolicyNet().to(device)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model.eval()

correct1 = 0
correct5 = 0
total = 0

# ======================================
# Evaluation
# ======================================

with torch.no_grad():

    for x, y in tqdm(loader, desc="Evaluating"):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        outputs = model(x)

        pred = outputs.argmax(dim=1)

        correct1 += (pred == y).sum().item()

        top5 = outputs.topk(5, dim=1).indices

        correct5 += (
            top5 == y.unsqueeze(1)
        ).any(dim=1).sum().item()

        total += y.size(0)

print(f"\nValidation positions : {total:,}")
print(f"Top-1 Accuracy       : {100 * correct1 / total:.2f}%")
print(f"Top-5 Accuracy       : {100 * correct5 / total:.2f}%")
