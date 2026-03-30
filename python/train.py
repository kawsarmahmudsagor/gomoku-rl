#!/usr/bin/env python3
"""
Main training script for Gomoku DQN agent.
Trains agent and exports model to ONNX format for web deployment.
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import GomokuEnv
from agent import DoubleDQNAgent
from opponents import RandomAgent, SelfPlayAgent
from training import GomokuTrainer


def export_to_onnx(agent: DoubleDQNAgent, output_path: str):
    """
    Export trained model to ONNX format for web deployment.

    Args:
        agent: Trained DoubleDQNAgent
        output_path: Path to save ONNX model
    """
    print(f"\nExporting model to ONNX format: {output_path}")

    # Create dummy input
    dummy_input = torch.randn(1, 1, 9, 9).to(agent.device)

    # Export to ONNX
    torch.onnx.export(
        agent.get_policy_network(),
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['q_values'],
        dynamic_axes={
            'board_state': {0: 'batch_size'},
            'q_values': {0: 'batch_size'}
        },
        verbose=False
    )

    print(f"✓ Model exported to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / 1e6:.2f} MB")


def export_model_weights(agent: DoubleDQNAgent, output_json: str):
    """
    Export model weights as JSON for reference/debugging.

    Args:
        agent: Trained DoubleDQNAgent
        output_json: Path to save JSON weights
    """
    print(f"\nExporting model weights as JSON: {output_json}")

    weights_dict = {}
    for name, param in agent.get_policy_network().named_parameters():
        weights_dict[name] = param.detach().cpu().numpy().tolist()

    import json
    with open(output_json, 'w') as f:
        json.dump(weights_dict, f, indent=2)

    print(f"✓ Weights exported to {output_json}")


def evaluate_agent(agent: DoubleDQNAgent, num_games: int = 100) -> float:
    """
    Evaluate trained agent's win rate against random opponent.

    Args:
        agent: Trained DoubleDQNAgent
        num_games: Number of games to play

    Returns:
        Win rate (0-1)
    """
    print(f"\nEvaluating agent ({num_games} games)...")

    opponent = RandomAgent()
    env = GomokuEnv()
    wins = 0

    for game_idx in range(num_games):
        state = env.reset()
        done = False

        while not done:
            # Agent move
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False, epsilon=0.0)
            state, _, done = env.step(action, player=1)

            if done:
                break

            # Opponent move
            opponent_action = opponent.get_action(state)
            state, _, done = env.step(opponent_action, player=-1)

        if env.winner == 1:
            wins += 1

        if (game_idx + 1) % 20 == 0:
            print(f"  Progress: {game_idx + 1}/{num_games} games")

    win_rate = wins / num_games
    print(f"✓ Evaluation complete: {wins}/{num_games} wins ({win_rate:.1%} win rate)")

    return win_rate


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train DQN agent for Gomoku")
    parser.add_argument("--episodes", type=int, default=200000,
                       help="Number of training episodes (default: 200000)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--gamma", type=float, default=0.99,
                       help="Discount factor (default: 0.99)")
    parser.add_argument("--buffer-size", type=int, default=100000,
                       help="Replay buffer size (default: 100000)")
    parser.add_argument("--checkpoint-interval", type=int, default=10000,
                       help="Save checkpoint every N episodes (default: 10000)")
    parser.add_argument("--eval-interval", type=int, default=1000,
                       help="Evaluate every N episodes (default: 1000)")
    parser.add_argument("--self-play-ratio", type=float, default=0.2,
                       help="Fraction of episodes against self-play (default: 0.2)")
    parser.add_argument("--no-export", action="store_true",
                       help="Skip ONNX export after training")
    parser.add_argument("--eval-only", action="store_true",
                       help="Skip training and only evaluate existing model")
    parser.add_argument("--model-path", type=str, default=None,
                       help="Path to existing model to load/evaluate")

    args = parser.parse_args()

    # Directories
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

    # Initialize agent
    agent = DoubleDQNAgent(
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_size=args.buffer_size
    )

    # Load existing model if provided
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        agent.load_model(args.model_path)

    # Evaluation only
    if args.eval_only:
        win_rate = evaluate_agent(agent, num_games=100)
        print(f"\nFinal win rate: {win_rate:.1%}")
        return

    # Initialize opponents
    random_opponent = RandomAgent(seed=42)
    self_play_agent = SelfPlayAgent()
    self_play_agent.set_model(agent)

    # Initialize trainer
    trainer = GomokuTrainer(
        agent=agent,
        random_opponent=random_opponent,
        self_play_agent=self_play_agent,
        model_dir=model_dir
    )

    # Train
    print("\n" + "="*60)
    print("🎮 GOMOKU RL TRAINING")
    print("="*60)
    print(f"Episodes: {args.episodes:,}")
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Gamma: {args.gamma}")
    print(f"Self-play Ratio: {args.self_play_ratio}")
    print(f"Device: {agent.device}")
    print("="*60 + "\n")

    trainer.train(
        num_episodes=args.episodes,
        checkpoint_interval=args.checkpoint_interval,
        eval_interval=args.eval_interval,
        self_play_ratio=args.self_play_ratio
    )

    # Save final model
    final_model_path = os.path.join(model_dir, "gomoku_agent_final.pt")
    agent.save_model(final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")

    # Evaluate final model
    win_rate = evaluate_agent(agent, num_games=100)

    # Export to ONNX (if not disabled)
    if not args.no_export:
        onnx_path = os.path.join(model_dir, "gomoku_agent.onnx")
        export_to_onnx(agent, onnx_path)

        json_path = os.path.join(model_dir, "gomoku_agent_weights.json")
        export_model_weights(agent, json_path)

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE")
    print("="*60)
    print(f"Final Win Rate: {win_rate:.1%}")
    print(f"Models saved to: {model_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
