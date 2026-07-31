import torch
import torch.nn as nn

from utils.move_vocab import VOCAB_SIZE


class ChessPolicyNet(nn.Module):

    def __init__(self, num_moves=VOCAB_SIZE):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8 → 4

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 4 → 2

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(256 * 2 * 2, 256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(256, num_moves)
        )

    def forward(self, x):

        x = self.features(x)

        return self.classifier(x)