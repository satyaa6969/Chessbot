import torch
import torch.nn as nn


class ChessPolicyNet(nn.Module):

    def __init__(self, num_moves=8192):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_moves)
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x