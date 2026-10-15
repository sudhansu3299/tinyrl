
# simulate.py
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac import SAC


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
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(
        args.env,
        render_mode="human" if args.render else None,
    )

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

    if metadata.get("state_dim", state_dim) != state_dim:
        raise ValueError("Observation dimension does not match checkpoint.")
    if metadata.get("action_dim", action_dim) != action_dim:
        raise ValueError("Action dimension does not match checkpoint.")

    action_scale = float(metadata.get("action_scale", 1.0))
    auto_entropy_tuning = bool(metadata.get("auto_entropy_tuning", False))

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        device=device,
        auto_entropy_tuning=auto_entropy_tuning,
    )
    agent.load_checkpoint(checkpoint_path, map_location=device)

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

    env.close()


if __name__ == "__main__":
    main()
