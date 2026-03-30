"""Self-play agent for Gomoku - uses a saved model"""

import numpy as np


class SelfPlayAgent:
    """Self-play agent that uses a saved DQN model for opponent moves."""

    def __init__(self, model=None):
        """
        Initialize self-play agent.

        Args:
            model: Trained DQN agent to use for moves
        """
        self.model = model

    def get_action(self, board: np.ndarray, valid_actions_mask: np.ndarray = None) -> int:
        """
        Select action using the trained model.

        Args:
            board: Current board state (9x9)
            valid_actions_mask: Optional mask of valid actions

        Returns:
            action: Action index selected by model
        """
        if self.model is None:
            raise RuntimeError("Model not set for self-play agent")

        # Use model to select action
        state = board.astype(np.float32).flatten()
        action = self.model.select_action(state, training=False, epsilon=0.0)
        return int(action)

    def set_model(self, model):
        """Set the model to use for action selection."""
        self.model = model

    def reset(self):
        """Reset agent state."""
        pass
