import torch
import torch.nn as nn
import torch.nn.functional as F

class AlphaZeroSimpleCNN(nn.Module):
    def __init__(self, sz=6, num_actions=36):
        super(AlphaZeroSimpleCNN, self).__init__()
        
        self.board_size = sz
        self.num_actions = num_actions

        # Convolutional block
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Two residual blocks for simplicity
        self.residual_block1 = self._build_residual_block(64)
        self.residual_block2 = self._build_residual_block(64)
        
        # Policy head
        self.policy_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1),
            nn.BatchNorm2d(2),
            nn.ReLU()
        )
        self.policy_fc = nn.Linear(2 * sz * sz, num_actions)  # Output size for policy logits
        
        # Value head
        self.value_conv = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.value_fc1 = nn.Linear(sz * sz, 64)  # Hidden layer for value
        self.value_fc2 = nn.Linear(64, 1)  # Scalar value output

    def _build_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        # x is (bsz, c, h, w)
        # Initial conv block
        x = self.conv_block(x)
        
        # Residual blocks
        residual = x
        x = self.residual_block1(x) + residual
        x = F.relu(x)
        
        residual = x
        x = self.residual_block2(x) + residual
        x = F.relu(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.policy_fc(policy)
        policy = torch.softmax(policy, dim=-1)
        
        # Value head
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output value in range [-1, 1]
        
        return policy.squeeze(), value.squeeze()