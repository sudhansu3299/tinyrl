
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
from action_visualizer import ActionVisualizer


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
    parser.add_argument("--update-freq", type=int, default=1)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--auto-entropy-tuning", action="store_true")
    parser.add_argument("--force-gpu", action="store_true", help="Fail if GPU is not available")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--visualize-actions", action="store_true", help="Show real-time action visualization")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env)
    env.action_space.seed(args.seed)

    # Detect if environment uses image observations (e.g., CarRacing-v3)
    use_cnn = args.env == "CarRacing-v3"
    
    if use_cnn:
        # For image observations, state_dim is the shape tuple (height, width, channels)
        state_dim = env.observation_space.shape
    else:
        state_dim = int(np.prod(env.observation_space.shape))
    
    action_dim = int(np.prod(env.action_space.shape))

    # Determine action scale based on environment
    if use_cnn:
        # CarRacing-v3 has action bounds [-1, 1] for steering and gas, [0, 1] for brake
        # We'll use action_scale=1.0 and handle scaling in the environment
        action_scale = 1.0
    else:
        # LunarLanderContinuous has symmetric action bounds [-1, 1].
        if not np.allclose(env.action_space.low, -1.0) or not np.allclose(
                env.action_space.high, 1.0
        ):
            raise ValueError(
                "This actor configuration expects symmetric action bounds [-1, 1]."
            )
        action_scale = 1.0

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using:", device)

    if args.force_gpu and device.type == "cpu":
        raise RuntimeError("--force-gpu specified but CUDA is not available. Please install PyTorch with CUDA support.")

    run_dir = Path(args.output_root) / args.env
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        device=device,
        auto_entropy_tuning=args.auto_entropy_tuning,
        use_cnn=use_cnn,
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=1_000_000,
        image_obs=use_cnn,
    )

    # Initialize action visualizer if requested
    visualizer = None
    if args.visualize_actions:
        visualizer = ActionVisualizer(action_dim, args.env, history_len=100)
        print(f"Action visualization enabled for {args.env}")

    csv_path = run_dir / "training_rewards.csv"
    checkpoint_path = run_dir / "latest.pth"
    best_path = run_dir / "best.pth"

    state, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_count = 0
    episode_start_time = None
    best_eval = -float("inf")

    with csv_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "step", "episode", "episode_return", "episode_length", "episode_time",
            "eval_mean", "eval_std"
        ])

        episode_start_time = None
        for step in range(1, args.steps + 1):
            if step == 1:
                episode_start_time = __import__('time').time()

            if step < args.learning_starts:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, evaluate=False)

            # Update action visualizer
            if visualizer is not None:
                visualizer.update(action)

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

            if step >= args.learning_starts and step % args.update_freq == 0:
                agent.update(replay_buffer, args.batch_size)

            if done:
                import time
                episode_time = time.time() - episode_start_time
                episode_count += 1
                print(
                    f"step={step} episode={episode_count} "
                    f"return={episode_return:.2f} length={episode_length} time={episode_time:.1f}s"
                )
                writer.writerow([
                    step, episode_count, episode_return, episode_length, f"{episode_time:.2f}", "", ""
                ])
                file.flush()

                state, _ = env.reset()
                episode_return = 0.0
                episode_length = 0
                episode_start_time = time.time()

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
                        "use_cnn": use_cnn,
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
                            "use_cnn": use_cnn,
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
            "use_cnn": use_cnn,
        },
    )

    # Close action visualizer if it was opened
    if visualizer is not None:
        visualizer.close()
        print("Action visualization closed")

    env.close()
    print(f"Saved checkpoint: {checkpoint_path}")
    print(f"Saved rewards:    {csv_path}")


if __name__ == "__main__":
    main()
