
# simulate.py
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac import SAC
from action_visualizer import ActionVisualizer
from wrappers import GrayscaleObservation, FrameStack


def make_env(env_id, render_mode=None):
    """Create an environment, optionally wrapping it for image-based tasks."""
    env = gym.make(env_id, render_mode=render_mode)
    use_cnn = env_id == "CarRacing-v3"
    if use_cnn:
        env = GrayscaleObservation(env)
        env = FrameStack(env, n=4)
    return env, use_cnn


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained SAC policy")
    parser.add_argument("--env", required=True, help="Gymnasium environment ID")
    parser.add_argument("--checkpoint", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastic actions instead of using the deterministic policy",
    )
    parser.add_argument("--visualize-actions", action="store_true", help="Show real-time action visualization")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env, use_cnn = make_env(
        args.env,
        render_mode="human" if args.render else None,
    )

    if use_cnn:
        # For image observations, state_dim is the shape tuple (height, width, channels)
        state_dim = env.observation_space.shape
    else:
        state_dim = int(np.prod(env.observation_space.shape))
    
    action_dim = int(np.prod(env.action_space.shape))

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    metadata = checkpoint.get("extra", {})

    if metadata.get("env_id") and metadata["env_id"] != args.env:
        raise ValueError(
            f"Checkpoint was trained on {metadata['env_id']}, "
            f"but --env specified {args.env}"
        )

    saved_state_dim = metadata.get("state_dim")
    if saved_state_dim is not None and saved_state_dim != state_dim:
        if use_cnn:
            # Old checkpoints used raw RGB (96,96,3); new ones use stacked grayscale (96,96,4)
            print(
                f"WARNING: Checkpoint state_dim {saved_state_dim} does not match "
                f"current environment {state_dim}. The checkpoint may have been trained "
                f"with a different observation format and may not load correctly."
            )
        else:
            raise ValueError("Observation dimension does not match checkpoint.")
    if metadata.get("action_dim", action_dim) != action_dim:
        raise ValueError("Action dimension does not match checkpoint.")

    action_scale = float(metadata.get("action_scale", 1.0))
    auto_entropy_tuning = bool(metadata.get("auto_entropy_tuning", False))
    use_cnn = bool(metadata.get("use_cnn", False))
    action_low = metadata.get("action_low", None)
    action_high = metadata.get("action_high", None)
    
    # Fallback: if checkpoint doesn't have action bounds, detect from environment
    if action_low is None and action_high is None:
        env_action_low = env.action_space.low.tolist()
        env_action_high = env.action_space.high.tolist()
        # Only use env bounds if they're asymmetric (not [-1, 1] for all dims)
        if not (env_action_low == [-1.0] * action_dim and env_action_high == [1.0] * action_dim):
            action_low = env_action_low
            action_high = env_action_high

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        device=device,
        auto_entropy_tuning=auto_entropy_tuning,
        use_cnn=use_cnn,
        action_low=action_low,
        action_high=action_high,
    )
    agent.load_checkpoint(checkpoint_path, map_location=device)

    # Initialize action visualizer if requested
    visualizer = None
    if args.visualize_actions:
        visualizer = ActionVisualizer(action_dim, args.env, history_len=100)
        print(f"Action visualization enabled for {args.env}")

    returns = []
    lengths = []

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed + episode)
        episode_return = 0.0
        length = 0
        done = False

        while not done:
            action = agent.select_action(
                state, evaluate=not args.stochastic
            )
            
            # Update action visualizer
            if visualizer is not None:
                visualizer.update(action)
            
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            length += 1
            done = terminated or truncated

        returns.append(episode_return)
        lengths.append(length)
        print(
            f"Episode {episode + 1:02d} | "
            f"Return: {episode_return:8.2f} | Steps: {length}"
        )

    print("\nEvaluation summary")
    print(f"Environment : {args.env}")
    print(f"Checkpoint  : {checkpoint_path}")
    print(f"Episodes    : {args.episodes}")
    print(f"Mean return : {np.mean(returns):.2f}")
    print(f"Std return  : {np.std(returns):.2f}")
    print(f"Mean length : {np.mean(lengths):.1f}")

    # Close action visualizer if it was opened
    if visualizer is not None:
        visualizer.close()
        print("Action visualization closed")

    env.close()


if __name__ == "__main__":
    main()
