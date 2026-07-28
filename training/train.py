import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training.dataset import ChessDataset
from training.model import ChessPolicyNet


# Hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
NUM_EPOCHS = 1


# Dataset
dataset = ChessDataset("data/processed/positions.csv")

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    num_workers=0
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


# Training loop
model.train()

for epoch in range(NUM_EPOCHS):

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
torch.save(model.state_dict(), "models/policy_net_epoch1.pth")
print("Model saved.")