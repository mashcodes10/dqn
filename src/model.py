"""
DQN CNN architecture from Mnih et al. (2015), Table 1.

Input:  4 x 84 x 84 (4 stacked grayscale frames)
Conv1:  32 filters, 8x8, stride 4 -> ReLU
Conv2:  64 filters, 4x4, stride 2 -> ReLU
Conv3:  64 filters, 3x3, stride 1 -> ReLU
FC1:    512 units -> ReLU
Output: one Q-value per action
"""

import torch
import torch.nn as nn


class DQN(nn.Module):
    def __init__(self, n_actions, in_channels=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # 64 * 7 * 7 = 3136 after convolutions on 84x84 input
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # x shape: (batch, channels, 84, 84), values in [0, 255]
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
