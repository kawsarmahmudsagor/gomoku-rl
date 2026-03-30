"""DQN Neural Network for Gomoku"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Gomoku.

    Converts 9x9 board state to 81 Q-values (one per action).
    Uses convolutional layers for spatial feature extraction.
    """

    def __init__(self, input_channels: int = 1, hidden_size: int = 64):
        """
        Initialize DQN network.

        Args:
            input_channels: Number of input channels (1 for single board)
            hidden_size: Number of channels in conv layers
        """
        super(DQNNetwork, self).__init__()

        self.board_size = 9
        self.actions = 81

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

        # Calculate size after conv layers (remains 9x9 with padding)
        conv_out_size = hidden_size * self.board_size * self.board_size

        # Fully connected layers
        self.fc1 = nn.Linear(conv_out_size, 256)
        self.fc2 = nn.Linear(256, self.actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.

        Args:
            x: Input tensor of shape (batch_size, channels, 9, 9)

        Returns:
            q_values: Q-values for each action, shape (batch_size, 81)
        """
        # Ensure input is float32
        x = x.float()

        # Convolutional layers with ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values


class DuelingDQNNetwork(nn.Module):
    """
    Dueling DQN Network for better learning.
    Separates value and advantage streams.
    """

    def __init__(self, input_channels: int = 1, hidden_size: int = 64):
        """Initialize Dueling DQN network."""
        super(DuelingDQNNetwork, self).__init__()

        self.board_size = 9
        self.actions = 81

        # Shared convolutional layers
        self.conv1 = nn.Conv2d(input_channels, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, kernel_size=3, padding=1)

        conv_out_size = hidden_size * self.board_size * self.board_size

        # Value stream
        self.value_fc = nn.Linear(conv_out_size, 128)
        self.value = nn.Linear(128, 1)

        # Advantage stream
        self.advantage_fc = nn.Linear(conv_out_size, 128)
        self.advantage = nn.Linear(128, self.actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through dueling network."""
        x = x.float()

        # Shared conv layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value(value)

        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)

        # Combine: Q = V + (A - mean(A))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values
