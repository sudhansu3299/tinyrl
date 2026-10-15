
# train.py
import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac import SAC
from replay_buffer import ReplayBuffer


def evaluate(agent, env_id, episodes, seed, render=False):
    render_mode = "human" if render else None
    env = gym.make(env_id, render_mode=render_mode)
    returns = []

    for episode in range(episodes):
        state, _ = env.reset(seed=seed + episode)
        episode_return = 0.0
        done = False

        while not done:
            # Deterministic action for evaluation.
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, _ = env.step(action)
            episode_return += float(reward)
            done = terminated or truncated

        returns.append(episode_return)

    env.close()
    return float(np.mean(returns)), float(np.std(returns)), returns


def main():
    parser = argparse.ArgumentParser(description="Train SAC")
    parser.add_argument("--env", default="LunarLanderContinuous-v3")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--auto-entropy-tuning", action="store_true")
    parser.add_argument("--output-root", default="runs")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))

    # LunarLanderContinuous has symmetric action bounds [-1, 1].
    if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
            env.action_space.high, 1.0
    ):
        raise ValueError(
            "This actor configuration expects symmetric action bounds [-1, 1]."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.output_root) / args.env
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=1.0,
        device=device,
        auto_entropy_tuning=args.auto_entropy_tuning,
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        # If your existing buffer constructor differs, keep its existing API.
        capacity=1_000_000,
    )

    csv_path = run_dir / "training_rewards.csv"
    checkpoint_path = run_dir / "latest.pth"
    best_path = run_dir / "best.pth"

    state, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    best_eval = -float("inf")

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "step", "episode", "episode_return", "episode_length",
            "eval_mean", "eval_std"
        ])

        for step in range(1, args.steps + 1):
            if step < args.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Use true terminal status for bootstrapping. A time-limit
            # truncation should not automatically be treated as terminal.
            replay_buffer.add(
                state, action, reward, next_state, float(terminated)
            )

            state = next_state
            episode_return += float(reward)
            episode_length += 1

            if step >= args.learning_starts:
                agent.update(replay_buffer, args.batch_size)

            if done:
                episode_count += 1
                print(
                    f"step={step} episode={episode_count} "
                    f"return={episode_return:.2f} length={episode_length}"
                )
                writer.writerow([
                    step, episode_count, episode_return, episode_length, "", ""
                ])
                file.flush()

                state, _ = env.reset()
                episode_return = 0.0
                episode_length = 0

            if step % args.eval_freq == 0:
                mean_ret, std_ret, _ = evaluate(
                    agent, args.env, args.eval_episodes,
                    args.seed + 10_000 + step
                )
                print(
                    f"[eval] step={step} "
                    f"return={mean_ret:.2f} +/- {std_ret:.2f}"
                )
                writer.writerow([
                    step, "", "", "", mean_ret, std_ret
                ])
                file.flush()

                # Save latest checkpoint; helper defined below in sac.py.
                agent.save_checkpoint(
                    checkpoint_path,
                    global_step=step,
                    extra={
                        "env_id": args.env,
                        "seed": args.seed,
                        "state_dim": state_dim,
                        "action_dim": action_dim,
                        "action_scale": 1.0,
                        "auto_entropy_tuning": args.auto_entropy_tuning,
                    },
                )

                if mean_ret > best_eval:
                    best_eval = mean_ret
                    agent.save_checkpoint(
                        best_path,
                        global_step=step,
                        extra={
                            "env_id": args.env,
                            "seed": args.seed,
                            "state_dim": state_dim,
                            "action_dim": action_dim,
                            "action_scale": 1.0,
                            "auto_entropy_tuning": args.auto_entropy_tuning,
                        },
                    )

    # Always save at the end, even if the final step isn't an eval step.
    agent.save_checkpoint(
        checkpoint_path,
        global_step=args.steps,
        extra={
            "env_id": args.env,
            "seed": args.seed,
            "state_dim": state_dim,
            "action_dim": action_dim,
            "action_scale": 1.0,
            "auto_entropy_tuning": args.auto_entropy_tuning,
        },
    )

    env.close()
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved rewards:    {csv_path}")


if __name__ == "__main__":
    main()
