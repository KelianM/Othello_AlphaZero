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
        
        # Value head
        value = self.value_conv(x)
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output value in range [-1, 1]
        
        # Return as numpy arrays
        return policy.squeeze().detach().numpy(), value.squeeze().detach().numpy()
    
def loss_function(pred_policy, pred_value, target_policy, target_value, mse_weight = 0.01):
    """
    Computes the loss for a batch of game examples.
    
    Arguments:
    - pred_policy: Tensor of shape [batch_size, num_actions], predicted policy p.
    - pred_value: Tensor of shape [batch_size, 1], predicted value z.
    - target_policy: Tensor of shape [batch_size, num_actions], target improved policy pi.
    - target_value: Tensor of shape [batch_size, 1], target value estimate v.
    - mse_weight: Value to weight the MSE loss by. Small to avoid overfitting to value.
    
    Returns:
    - total_loss: The combined value loss and policy loss for the batch.
    """
    # Value loss: Mean Squared Error (MSE) between predicted value and target value
    value_loss = F.mse_loss(pred_value, target_value)
    
    # Policy loss: Cross-entropy (negative log likelihood) between target policy and predicted policy
    policy_loss = -(target_policy * torch.log(pred_policy + 1e-10)).sum(dim=1).mean()
    
    # Total loss is the sum of value loss and policy loss
    total_loss = mse_weight*value_loss + policy_loss
    
    return total_loss