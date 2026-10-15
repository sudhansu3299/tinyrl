
import csv
import random
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from sac import SAC
from replay_buffer import ReplayBuffer

import argparse


ENV_NAME = "Pendulum-v1"
SEEDS = [42, 123, 7]
TOTAL_STEPS = 50_000
LEARNING_STARTS = 5_000
BATCH_SIZE = 256
EVAL_INTERVAL = 5_000
EVAL_EPISODES = 5

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def evaluate(agent, env, episodes=5):
    returns = []

    for episode in range(episodes):
        state, _ = env.reset(seed=10_000 + episode)
        total_return = 0.0
        done = False

        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            total_return += reward
            done = terminated or truncated

        returns.append(total_return)

    return float(np.mean(returns)), float(np.std(returns))


def run_experiment(seed, auto_alpha, writer):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = gym.make(ENV_NAME)
    eval_env = gym.make(ENV_NAME)
    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = float(env.action_space.high[0])

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        lr=3e-4,
        auto_entropy_tuning=auto_alpha,
    )

    buffer = ReplayBuffer(
        state_dim, action_dim, capacity=200_000
    )

    state, _ = env.reset(seed=seed)
    episode_return = 0.0
    episode_num = 0

    mode = "automatic" if auto_alpha else "fixed"
    print(f"\nStarting {mode} alpha | seed={seed}")

    for step in range(1, TOTAL_STEPS + 1):
        if step <= LEARNING_STARTS:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state)

        next_state, reward, terminated, truncated, _ = env.step(action)

        buffer.add(
            state, action, reward, next_state, float(terminated)
        )

        state = next_state
        episode_return += reward

        if step >= LEARNING_STARTS and buffer.size >= BATCH_SIZE:
            agent.update(buffer, BATCH_SIZE)

        if terminated or truncated:
            episode_num += 1

            print(
                f"{mode} | seed={seed} | step={step} | "
                f"episode={episode_num} | return={episode_return:.1f}"
            )

            state, _ = env.reset()
            episode_return = 0.0

        if step % EVAL_INTERVAL == 0:
            mean_return, std_return = evaluate(
                agent, eval_env, EVAL_EPISODES
            )

            row = {
                "mode": mode,
                "seed": seed,
                "step": step,
                "eval_mean": mean_return,
                "eval_std": std_return,
                "alpha": agent.alpha,
            }
            writer.writerow(row)

            print(
                f"Evaluation: {mode}, seed={seed}, step={step}, "
                f"return={mean_return:.2f} +/- {std_return:.2f}, "
                f"alpha={agent.alpha:.4f}"
            )

    env.close()
    eval_env.close()


def plot_results():
    import pandas as pd

    df = pd.read_csv(RESULTS_DIR / "sac_comparison.csv")

    summary = (
        df.groupby(["mode", "step"])["eval_mean"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for mode in summary["mode"].unique():
        data = summary[summary["mode"] == mode].sort_values("step")

        # Across-seed variability, not evaluation-episode variability.
        std = data["std"].fillna(0)

        plt.plot(data["step"], data["mean"], marker="o", label=mode)
        plt.fill_between(
            data["step"],
            data["mean"] - std,
            data["mean"] + std,
            alpha=0.2,
            )

    plt.xlabel("Environment steps")
    plt.ylabel("Mean evaluation return")
    plt.title("SAC: Fixed vs Automatic Entropy Tuning")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "sac_comparison.png", dpi=150)
    plt.show()

    print("\nFinal evaluation summary:")
    print(
        summary[summary["step"] == TOTAL_STEPS]
        .groupby("mode")["mean"]
        .agg(["mean", "std"])
    )



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["fixed", "automatic", "both"],
        default="both",
        help="Which SAC entropy mode to run",
    )
    args = parser.parse_args()

    modes = {
        "fixed": [False],
        "automatic": [True],
        "both": [False, True],
    }[args.mode]

    with open(
            RESULTS_DIR / "sac_comparison.csv",
            "w",
            newline="",
    ) as f:
        fieldnames = [
            "mode", "seed", "step",
            "eval_mean", "eval_std", "alpha",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for auto_alpha in modes:
            for seed in SEEDS:
                run_experiment(seed, auto_alpha, writer)
                f.flush()

    plot_results()

if __name__ == "__main__":
    main()
