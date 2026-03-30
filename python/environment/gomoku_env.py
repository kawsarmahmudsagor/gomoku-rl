"""Gomoku Environment - 9x9 Board Game with RL interface"""

import numpy as np
from typing import Tuple, Dict, List


class GomokuEnv:
    """
    9x9 Gomoku (Five in a row) game environment.

    State encoding:
    - 0: empty cell
    - 1: agent stone
    - -1: opponent stone

    Actions: 0-80 (left-to-right, top-to-bottom)
    """

    BOARD_SIZE = 9
    WIN_LENGTH = 5

    def __init__(self):
        """Initialize Gomoku environment."""
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.move_history = []
        self.game_over = False
        self.winner = None  # None, 1 (agent), -1 (opponent), 0 (draw)

    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        Returns:
            state: 9x9 board state
        """
        self.board = np.zeros((self.BOARD_SIZE, self.BOARD_SIZE), dtype=np.int8)
        self.move_history = []
        self.game_over = False
        self.winner = None
        return self.board.copy()

    def action_to_pos(self, action: int) -> Tuple[int, int]:
        """Convert action index (0-80) to board position (row, col)."""
        row = action // self.BOARD_SIZE
        col = action % self.BOARD_SIZE
        return row, col

    def pos_to_action(self, row: int, col: int) -> int:
        """Convert board position to action index."""
        return row * self.BOARD_SIZE + col

    def is_valid(self, action: int) -> bool:
        """Check if action is valid (cell empty and action in range)."""
        if action < 0 or action >= self.BOARD_SIZE * self.BOARD_SIZE:
            return False
        row, col = self.action_to_pos(action)
        return self.board[row, col] == 0

    def get_valid_actions(self) -> np.ndarray:
        """Get array of all valid actions (0=invalid, 1=valid)."""
        valid = np.ones(self.BOARD_SIZE * self.BOARD_SIZE, dtype=np.int8)
        for action in range(self.BOARD_SIZE * self.BOARD_SIZE):
            if not self.is_valid(action):
                valid[action] = 0
        return valid

    def check_winner(self, player: int) -> bool:
        """Check if player has won (5 consecutive stones)."""
        board = self.board

        # Check rows
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
                if all(board[row, col + i] == player for i in range(self.WIN_LENGTH)):
                    return True

        # Check columns
        for col in range(self.BOARD_SIZE):
            for row in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
                if all(board[row + i, col] == player for i in range(self.WIN_LENGTH)):
                    return True

        # Check diagonals (top-left to bottom-right)
        for row in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
            for col in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
                if all(board[row + i, col + i] == player for i in range(self.WIN_LENGTH)):
                    return True

        # Check diagonals (top-right to bottom-left)
        for row in range(self.BOARD_SIZE - self.WIN_LENGTH + 1):
            for col in range(self.WIN_LENGTH - 1, self.BOARD_SIZE):
                if all(board[row + i, col - i] == player for i in range(self.WIN_LENGTH)):
                    return True

        return False

    def is_game_over(self) -> bool:
        """Check if game is over (winner or board full)."""
        # Check for winners
        if self.check_winner(1):
            self.winner = 1
            return True
        if self.check_winner(-1):
            self.winner = -1
            return True

        # Check if board is full
        if not np.any(self.board == 0):
            self.winner = 0  # Draw
            return True

        return False

    def step(self, action: int, player: int) -> Tuple[np.ndarray, float, bool]:
        """
        Execute one step of the environment.

        Args:
            action: Action index (0-80)
            player: Player making move (1 for agent, -1 for opponent)

        Returns:
            next_state: New board state
            reward: Immediate reward
            done: Whether game is over
        """
        if self.game_over:
            raise RuntimeError("Game is already over. Call reset() to start a new game.")

        # Validate action
        if not self.is_valid(action):
            return self.board.copy(), -0.5, False  # Invalid move penalty

        # Place stone
        row, col = self.action_to_pos(action)
        self.board[row, col] = player
        self.move_history.append((action, player))

        # Check game over
        done = self.is_game_over()

        # Determine reward
        reward = -0.01  # Small cost per move

        if done:
            if self.winner == player:
                reward = 1.0  # Win
            elif self.winner == 0:
                reward = 0.0  # Draw
            else:
                reward = -1.0  # Loss
            self.game_over = True

        return self.board.copy(), reward, done

    def get_state_for_agent(self, player: int) -> np.ndarray:
        """
        Get board state from player's perspective.
        Agent always sees itself as 1, opponent as -1.
        """
        if player == 1:
            return self.board.copy().astype(np.float32)
        else:
            # Flip perspective for opponent (multiply by -1)
            return (-self.board).copy().astype(np.float32)

    def render(self) -> str:
        """Return ASCII representation of board."""
        symbols = {0: "·", 1: "●", -1: "○"}
        lines = ["  0 1 2 3 4 5 6 7 8"]
        for row in range(self.BOARD_SIZE):
            line = f"{row} "
            for col in range(self.BOARD_SIZE):
                line += symbols[self.board[row, col]] + " "
            lines.append(line)
        return "\n".join(lines)

    def get_game_info(self) -> Dict:
        """Get current game information."""
        return {
            "board": self.board.copy(),
            "game_over": self.game_over,
            "winner": self.winner,
            "move_count": len(self.move_history),
            "valid_moves": np.sum(self.get_valid_actions())
        }
