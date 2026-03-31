"""Training loop and metrics for Gomoku RL agent"""

import numpy as np
from tqdm import tqdm
from datetime import datetime
import json
import os
from typing import Dict, List, Tuple

from environment import GomokuEnv
from agent import DoubleDQNAgent
from opponents import RandomAgent, SelfPlayAgent


class GomokuTrainer:
    """
    Trainer class for DQN agent playing Gomoku.
    Handles episode execution, metrics tracking, and checkpointing.
    """

    def __init__(self,
                 agent: DoubleDQNAgent,
                 random_opponent: RandomAgent,
                 self_play_agent: SelfPlayAgent = None,
                 model_dir: str = "python/models",
                 log_dir: str = "python/models/logs"):
        """
        Initialize trainer.

        Args:
            agent: DoubleDQNAgent to train
            random_opponent: Random opponent for training
            self_play_agent: Optional self-play opponent
            model_dir: Directory to save models
            log_dir: Directory to save training logs
        """
        self.agent = agent
        self.random_opponent = random_opponent
        self.self_play_agent = self_play_agent
        self.model_dir = model_dir
        self.log_dir = log_dir

        # Create directories
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rate_history = []
        self.loss_history = []
        self.checkpoint_rewards = []

    def play_episode(self,
                    opponent_type: str = "random",
                    training: bool = True) -> Dict:
        """
        Play one episode of Gomoku.

        Args:
            opponent_type: "random" or "self_play"
            training: Whether to update agent weights

        Returns:
            Dictionary with episode statistics
        """
        env = GomokuEnv()
        state = env.reset()

        episode_reward = 0
        episode_loss = 0
        loss_count = 0
        done = False
        step = 0

        while not done and step < 81:  # Max 81 moves on 9x9 board
            # Agent's move (always player 1)
            valid_actions = env.get_valid_actions()
            action = self.agent.select_action(state, valid_actions, training=training)

            next_state, reward, done = env.step(action, player=1)
            episode_reward += reward

            # Store experience
            if training:
                self.agent.add_experience(state, action, reward, next_state, done)
                loss = self.agent.train_step(batch_size=32)
                if loss is not None:
                    episode_loss += loss
                    loss_count += 1

            if done:
                break

            state = next_state
            step += 1

            # Opponent's move (player -1)
            valid_actions = env.get_valid_actions()
            if opponent_type == "self_play" and self.agent is not None:
                opponent_action = self.self_play_agent.get_action(state, valid_actions)
            else:
                opponent_action = self.random_opponent.get_action(state)

            next_state, opp_reward, done = env.step(opponent_action, player=-1)

            # From agent's perspective, opponent's reward is negative
            opponent_reward = -opp_reward
            episode_reward += opponent_reward

            if training:
                self.agent.add_experience(state, opponent_action, opponent_reward,
                                         next_state, done)

            state = next_state
            step += 1

        # Calculate loss
        avg_loss = episode_loss / max(loss_count, 1)

        # Determine game result
        if env.winner == 1:
            result = "win"
        elif env.winner == -1:
            result = "loss"
        else:
            result = "draw"

        return {
            "reward": episode_reward,
            "steps": step,
            "result": result,
            "avg_loss": avg_loss,
            "epsilon": self.agent._get_epsilon()
        }

    def train(self,
             num_episodes: int = 400000,
             checkpoint_interval: int = 20000,
             eval_interval: int = 2000,
             self_play_ratio: float = 0.2):
        """
        Main training loop.

        Args:
            num_episodes: Total episodes to train
            checkpoint_interval: Save model every N episodes
            eval_interval: Evaluate every N episodes
            self_play_ratio: Fraction of episodes against self-play
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Self-play ratio: {self_play_ratio}")

        # Training loop
        for episode in tqdm(range(num_episodes), desc="Training"):
            # Choose opponent
            if self.self_play_agent is not None and np.random.random() < self_play_ratio:
                result = self.play_episode(opponent_type="self_play", training=True)
            else:
                result = self.play_episode(opponent_type="random", training=True)

            self.episode_rewards.append(result["reward"])
            self.episode_lengths.append(result["steps"])
            self.loss_history.append(result["avg_loss"])

            # Periodic evaluation
            if (episode + 1) % eval_interval == 0:
                win_count = sum(1 for r in self.episode_rewards[-eval_interval:]
                              if isinstance(r, dict) or
                              (self.episode_rewards[-eval_interval:][
                                  self.episode_rewards[-eval_interval:].index(r)
                              ] > 0))

                # Track recent results
                recent_results = self.get_recent_results(eval_interval)
                win_rate = recent_results["win_rate"]

                tqdm.write(f"Episode {episode+1}: Win Rate = {win_rate:.2%}, "
                          f"Avg Reward = {np.mean(self.episode_rewards[-eval_interval:]):.3f}, "
                          f"Epsilon = {result['epsilon']:.4f}")

                self.win_rate_history.append({
                    "episode": episode + 1,
                    "win_rate": win_rate
                })

            # Checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                self.save_checkpoint(episode + 1)
                print(f"Checkpoint saved at episode {episode + 1}")

        print("Training completed!")
        self.save_final_metrics()

    def get_recent_results(self, window: int = 1000) -> Dict:
        """Get statistics from recent episodes."""
        if not self.episode_rewards:
            return {"win_rate": 0, "avg_reward": 0}

        recent = self.episode_rewards[-window:]
        win_count = min(len(recent), window)  # Simplified - real tracking in play_episode

        # More accurate win rate from last eval episodes
        env = GomokuEnv()
        wins = 0
        for _ in range(min(100, window)):  # Eval on 100 games
            state = env.reset()
            done = False
            while not done:
                valid_actions = env.get_valid_actions()
                action = self.agent.select_action(state, valid_actions, training=False)
                state, _, done = env.step(action, 1)
                if done:
                    break

                valid_actions = env.get_valid_actions()
                opp_action = self.random_opponent.get_action(state)
                state, _, done = env.step(opp_action, -1)

            if env.winner == 1:
                wins += 1

        return {
            "win_rate": wins / 100.0,
            "avg_reward": np.mean(recent)
        }

    def save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.model_dir, f"agent_ep{episode}.pt")
        self.agent.save_model(checkpoint_path)

    def save_final_metrics(self):
        """Save training metrics to file."""
        metrics = {
            "episode_rewards": self.episode_rewards[-10000:],  # Last 10k
            "episode_lengths": self.episode_lengths[-10000:],
            "loss_history": self.loss_history[-10000:],
            "win_rate_history": self.win_rate_history,
            "timestamp": datetime.now().isoformat()
        }

        metrics_path = os.path.join(self.log_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Metrics saved to {metrics_path}")
