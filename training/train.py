import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from training.dataset import ChessDataset
from training.model import ChessPolicyNet


# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1

CHECKPOINT_DIR = "/kaggle/working/models"
CHECKPOINT_FILE = os.path.join(CHECKPOINT_DIR, "latest_checkpoint.pth")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
# Dataset
dataset = ChessDataset("/kaggle/input/datasets/satyajeet69/positions-csv/positions.csv")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    pin_memory=True,
)


# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ChessPolicyNet().to(device)


# Loss
criterion = nn.CrossEntropyLoss()


# Optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

start_epoch = 0

if os.path.exists(CHECKPOINT_FILE):
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")
else:
    print("No checkpoint found. Starting fresh.")
# Training loop
model.train()

for epoch in range(start_epoch, NUM_EPOCHS):

    running_loss = 0.0

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(x)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if batch_idx % 1000 == 0:

            avg_loss = running_loss / (batch_idx + 1)
            positions_seen = (batch_idx + 1) * BATCH_SIZE

            print(
                f"Epoch {epoch + 1} | "
                f"Positions: {positions_seen:,} | "
                f"Loss: {avg_loss:.4f}"
            )
            print(
                f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx}] "
                f"Loss: {avg_loss:.4f}"
            )
        if batch_idx % 5000 == 0 and batch_idx != 0:
            torch.save(
                model.state_dict(),
                f"/kaggle/working/models/checkpoint_{batch_idx}.pth"
            )
            print(f"Checkpoint saved at batch {batch_idx}")
torch.save(
    model.state_dict(),
    "/kaggle/working/models/policy_net_epoch1.pth"
)
print("Model saved.")