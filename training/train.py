import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import ChessDataset
from training.model import ChessPolicyNet

# ===========================
# Configuration
# ===========================

TRAIN_PATH = "/kaggle/input/datasets/satyajeet69/training-set/train.csv"

MODEL_DIR = "/kaggle/working/models"
CHECKPOINT_FILE = os.path.join(MODEL_DIR, "latest_checkpoint.pth")
FINAL_MODEL = os.path.join(MODEL_DIR, "policy_net_epoch1.pth")

BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1
CHECKPOINT_INTERVAL = 5000
LOG_INTERVAL = 1000

os.makedirs(MODEL_DIR, exist_ok=True)

# ===========================
# Dataset
# ===========================

dataset = ChessDataset(TRAIN_PATH)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0,
    pin_memory=True,
)

# ===========================
# Device
# ===========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

# ===========================
# Model
# ===========================

model = ChessPolicyNet().to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=len(loader) * NUM_EPOCHS
)

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

start_epoch = 0

# ===========================
# Resume Training
# ===========================

if os.path.exists(CHECKPOINT_FILE):

    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = checkpoint["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")

else:

    print("No checkpoint found. Starting fresh.")

# ===========================
# Training
# ===========================

model.train()

training_start = time.time()

for epoch in range(start_epoch, NUM_EPOCHS):

    running_loss = 0.0

    epoch_start = time.time()

    for batch_idx, (x, y) in enumerate(loader):

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):

            outputs = model(x)

            loss = criterion(outputs, y)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        running_loss += loss.item()

        if batch_idx % LOG_INTERVAL == 0:

            avg_loss = running_loss / (batch_idx + 1)

            elapsed = time.time() - epoch_start

            lr = scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
                f"Batch {batch_idx:,}/{len(loader):,} | "
                f"Loss {avg_loss:.4f} | "
                f"LR {lr:.6f} | "
                f"Time {elapsed:.1f}s"
            )

        if batch_idx % CHECKPOINT_INTERVAL == 0 and batch_idx != 0:

            torch.save({

                "epoch": epoch,

                "model_state": model.state_dict(),

                "optimizer_state": optimizer.state_dict(),

                "scheduler_state": scheduler.state_dict(),

            }, CHECKPOINT_FILE)

            print("Checkpoint updated.")

# ===========================
# Save Final Model
# ===========================

torch.save(model.state_dict(), FINAL_MODEL)

print(f"Final model saved to:\n{FINAL_MODEL}")

print(
    f"Training completed in "
    f"{(time.time() - training_start)/60:.2f} minutes."
)