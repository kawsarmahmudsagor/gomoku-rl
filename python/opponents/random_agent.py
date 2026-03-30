"""Random opponent agent for Gomoku"""

import numpy as np
from typing import Tuple


class RandomAgent:
    """Random player that selects uniformly from valid moves."""

    def __init__(self, seed: int = None):
        """
        Initialize random agent.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def get_action(self, board: np.ndarray) -> int:
        """
        Select a random valid action.

        Args:
            board: Current board state (9x9)

        Returns:
            action: Random valid action index (0-80)
        """
        valid_actions = np.where(board.flatten() == 0)[0]
        if len(valid_actions) == 0:
            raise RuntimeError("No valid moves available")
        return np.random.choice(valid_actions)

    def reset(self):
        """Reset agent state (if needed)."""
        pass
