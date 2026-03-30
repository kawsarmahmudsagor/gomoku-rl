"""Double DQN Agent for Gomoku"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple

from .network import DQNNetwork
from .experience_replay import ExperienceReplayBuffer


class DoubleDQNAgent:
    """
    Double DQN agent for learning to play Gomoku.
    Uses target network to reduce overestimation bias.
    """

    def __init__(self,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05,
                 epsilon_decay: float = 500000,
                 tau: float = 0.001,
                 buffer_size: int = 100000,
                 device: str = None):
        """
        Initialize Double DQN agent.

        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Initial exploration rate
            epsilon_end: Final exploration rate
            epsilon_decay: Steps to decay epsilon
            tau: Soft update parameter for target network
            buffer_size: Replay buffer capacity
            device: Device to use (cuda/cpu)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Hyperparameters
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.learning_rate = learning_rate

        # Networks
        self.policy_network = DQNNetwork().to(self.device)
        self.target_network = DQNNetwork().to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_network.parameters(),
                                   lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss

        # Replay buffer
        self.replay_buffer = ExperienceReplayBuffer(capacity=buffer_size)

        # Training step counter
        self.steps = 0

    def select_action(self,
                     state: np.ndarray,
                     valid_actions_mask: np.ndarray = None,
                     training: bool = True,
                     epsilon: float = None) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Board state (flattened or 2D)
            valid_actions_mask: Mask of valid actions (81,)
            training: Whether in training mode
            epsilon: Override epsilon value

        Returns:
            action: Selected action index
        """
        if epsilon is None:
            epsilon = self._get_epsilon() if training else 0.0

        # Epsilon-greedy
        if np.random.random() < epsilon:
            # Explore: select random valid action
            if valid_actions_mask is not None:
                valid_actions = np.where(valid_actions_mask)[0]
            else:
                valid_actions = np.arange(81)

            if len(valid_actions) == 0:
                return np.random.randint(0, 81)
            return np.random.choice(valid_actions).item()

        else:
            # Exploit: select best action according to policy network
            # Ensure state is properly shaped
            if state.ndim == 1:
                state = state.reshape(1, 1, 9, 9)
            elif state.ndim == 2:
                state = np.expand_dims(np.expand_dims(state, 0), 0)
            else:
                state = np.expand_dims(state, 0)

            state_tensor = torch.FloatTensor(state).to(self.device)

            with torch.no_grad():
                q_values = self.policy_network(state_tensor)
                q_values = q_values.cpu().numpy()[0]

            # Apply valid actions mask
            if valid_actions_mask is not None:
                q_values = q_values * valid_actions_mask
                q_values[valid_actions_mask == 0] = -np.inf

            action = np.argmax(q_values).item()
            return action

    def train_step(self, batch_size: int = 32):
        """
        Perform one training step using batch from replay buffer.

        Args:
            batch_size: Size of batch to train on

        Returns:
            loss: Loss value
        """
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Debug: Print shapes before reshaping
        if self.steps % 1000 == 0:
            print(f"DEBUG - States shape before reshape: {states.shape}")

        # Reshape states for network - ensure (batch, 1, 9, 9)
        if states.ndim == 2:
            states = states.view(-1, 1, 9, 9)
            next_states = next_states.view(-1, 1, 9, 9)
        elif states.ndim == 3:
            # Already (batch, 9, 9) - add channel dimension using view instead of unsqueeze
            batch = states.shape[0]
            states = states.view(batch, 1, 9, 9)
            next_states = next_states.view(batch, 1, 9, 9)
            if self.steps % 1000 == 0:
                print(f"DEBUG - States shape after reshape: {states.shape}")

        # Current Q-values
        current_q = self.policy_network(states)
        current_q = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Next Q-values (using Double DQN)
        with torch.no_grad():
            # Policy network selects best action
            next_q_policy = self.policy_network(next_states)
            next_actions = next_q_policy.argmax(dim=1)

            # Target network evaluates the action
            next_q_target = self.target_network(next_states)
            next_q_max = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Bellman target
            target_q = rewards + (1 - dones) * self.gamma * next_q_max

        # Calculate loss
        loss = self.loss_fn(current_q, target_q)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 10.0)
        self.optimizer.step()

        # Soft update target network
        self._soft_update()

        self.steps += 1

        return loss.item()

    def _soft_update(self):
        """Soft update target network: θ' = τ*θ + (1-τ)*θ'"""
        for target_param, policy_param in zip(self.target_network.parameters(),
                                             self.policy_network.parameters()):
            target_param.data.copy_(
                self.tau * policy_param.data + (1.0 - self.tau) * target_param.data
            )

    def _get_epsilon(self) -> float:
        """Calculate current epsilon based on training steps."""
        epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                  np.exp(-1.0 * self.steps / self.epsilon_decay)
        return epsilon

    def add_experience(self, state: np.ndarray, action: int, reward: float,
                      next_state: np.ndarray, done: bool):
        """Add experience to replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)

    def save_model(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        print(f"Model loaded from {path}")

    def get_policy_network(self):
        """Return policy network for inference."""
        return self.policy_network
