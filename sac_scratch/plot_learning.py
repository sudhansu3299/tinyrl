
# plot_learning.py
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Plot SAC episode rewards and moving average"
    )
    parser.add_argument(
        "--csv",
        default="runs/LunarLanderContinuous-v3/training_rewards.csv",
    )
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--output", default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    episodes = []
    rewards = []

    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("episode_return", "").strip():
                episodes.append(int(row["episode"]))
                rewards.append(float(row["episode_return"]))

    if not rewards:
        raise ValueError(
            "No training episode rewards found in the CSV. "
            "Check that training_rewards.csv contains episode_return values."
        )

    rewards = np.asarray(rewards, dtype=float)
    episodes = np.asarray(episodes)

    if args.window < 1:
        raise ValueError("--window must be >= 1")

    moving_average = np.convolve(
        rewards,
        np.ones(args.window) / args.window,
        mode="valid",
        )
    moving_episodes = episodes[args.window - 1:]

    plt.figure(figsize=(12, 6))

    # Raw episode returns: noisy continuous line.
    plt.plot(
        episodes,
        rewards,
        label="Episode return",
        alpha=0.45,
        linewidth=1,
    )

    # Smoothed trend: one point for each full rolling window.
    plt.plot(
        moving_episodes,
        moving_average,
        label=f"{args.window}-episode moving average",
        linewidth=2.5,
    )

    env_name = csv_path.parent.name
    plt.title(f"SAC Learning Curve — {env_name}")
    plt.xlabel("Episode")
    plt.ylabel("Episode return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    output_path = Path(
        args.output
        or csv_path.parent / "learning_curve.png"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    print(f"Saved plot to: {output_path}")
    print(f"Training episodes plotted: {len(rewards)}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
