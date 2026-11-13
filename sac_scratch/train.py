
# train.py
import argparse
import csv
import random
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sac import SAC
from replay_buffer import ReplayBuffer, MixedReplayBuffer
from action_visualizer import ActionVisualizer
from wrappers import GrayscaleObservation, FrameStack


def _save_checkpoint_with_buffer(agent, replay_buffer, checkpoint_path, replay_buffer_path,
                                  step, args, state_dim, action_dim, use_cnn,
                                  action_low, action_high, best_eval, episode_count):
    """Save agent checkpoint along with replay buffer and training metadata."""
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
            "action_low": action_low,
            "action_high": action_high,
            "best_eval": best_eval,
            "episode_count": episode_count,
        },
    )
    replay_buffer.save(str(replay_buffer_path))


def make_env(env_id, render_mode=None):
    """Create an environment, optionally wrapping it for image-based tasks."""
    env = gym.make(env_id, render_mode=render_mode)
    use_cnn = env_id == "CarRacing-v3"
    if use_cnn:
        env = GrayscaleObservation(env)
        env = FrameStack(env, n=4)
    return env, use_cnn


def evaluate(agent, env_id, episodes, seed, render=False):
    env, _ = make_env(env_id, render_mode="human" if render else None)
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
    parser.add_argument("--gradient-steps", type=int, default=None,
                        help="Gradient steps per env step (default: 1 for non-CNN, 1 for CNN; increase to saturate GPU)")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--auto-entropy-tuning", action="store_true")
    parser.add_argument("--force-gpu", action="store_true", help="Fail if GPU is not available")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--visualize-actions", action="store_true", help="Show real-time action visualization")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume training from")
    parser.add_argument("--checkpoint-freq", type=int, default=50_000, help="Save checkpoint every N steps (default: 50000)")
    parser.add_argument("--demo-dir", type=str, default=None,
                        help="Comma-separated dirs/files of recorded CarRacing demos (.npz)")
    parser.add_argument("--demo-ratio", type=float, default=0.3,
                        help="Fraction of each update batch drawn from demos (default: 0.3)")
    parser.add_argument("--demo-stride", type=int, default=1,
                        help="Use every Nth demo transition (default: 1)")
    parser.add_argument("--demo-max-transitions", type=int, default=50_000,
                        help="Cap on demo transitions loaded into memory (default: 50000, ~1.8 GB)")
    parser.add_argument("--demo-capacity", type=int, default=300_000,
                        help="Capacity of the demonstration replay buffer (default: 300000)")
    parser.add_argument("--bc-checkpoint", type=str, default=None,
                        help="Actor checkpoint from pretrain_bc.py to warm-start from")
    args = parser.parse_args()

    if args.resume and args.bc_checkpoint:
        parser.error("--resume and --bc-checkpoint are mutually exclusive "
                     "(the resumed checkpoint already contains the actor).")
    if args.demo_ratio < 0.0 or args.demo_ratio > 1.0:
        parser.error("--demo-ratio must be in [0, 1]")
    if args.demo_dir and args.env != "CarRacing-v3":
        parser.error("--demo-dir is only supported for CarRacing-v3 image observations.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env, use_cnn = make_env(args.env)
    env.action_space.seed(args.seed)

    # Detect if environment uses image observations (e.g., CarRacing-v3)
    # use_cnn is already set by make_env based on env_id
    
    if use_cnn:
        # For image observations, state_dim is the shape tuple (height, width, channels)
        # With frame stacking: (96, 96, 4) — 4 grayscale frames
        state_dim = env.observation_space.shape
        # CarRacing-v3 action bounds: steering [-1, 1], gas [0, 1], brake [0, 1]
        action_low = env.action_space.low.tolist()
        action_high = env.action_space.high.tolist()
    else:
        state_dim = int(np.prod(env.observation_space.shape))
        action_low = None
        action_high = None
    
    action_dim = int(np.prod(env.action_space.shape))

    # Determine action scale based on environment
    if use_cnn:
        # CarRacing-v3 uses per-dimension action bounds, action_scale is ignored
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

    batch_size = args.batch_size
    gradient_steps = args.gradient_steps if args.gradient_steps is not None else 1

    print(f"Batch size: {batch_size}, Gradient steps: {gradient_steps}")

    run_dir = Path(args.output_root) / args.env
    run_dir.mkdir(parents=True, exist_ok=True)

    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=action_scale,
        device=device,
        auto_entropy_tuning=args.auto_entropy_tuning,
        use_cnn=use_cnn,
        action_low=action_low,
        action_high=action_high,
    )

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=1_000_000,
        image_obs=use_cnn,
    )

    # Demo-augmented training: keep a separate buffer of expert transitions and
    # draw a fixed fraction of every update batch from it.
    if args.demo_dir:
        from demo_utils import DemoDataset, fill_buffer_from_demos, load_episodes

        demo_paths = [p.strip() for p in args.demo_dir.split(",") if p.strip()]
        demo_episodes = load_episodes(demo_paths)
        demo_dataset = DemoDataset(
            demo_episodes,
            n_stack=4,
            stride=args.demo_stride,
            max_transitions=args.demo_max_transitions,
        )
        if tuple(demo_dataset.state_shape) != tuple(state_dim):
            raise ValueError(
                f"Demo state shape {demo_dataset.state_shape} does not match "
                f"environment observation space {state_dim}."
            )

        demo_buffer = ReplayBuffer(
            state_dim=state_dim,
            action_dim=action_dim,
            capacity=max(args.demo_capacity, len(demo_dataset)),
            image_obs=True,
        )
        n_demos = fill_buffer_from_demos(demo_buffer, demo_dataset)
        replay_buffer = MixedReplayBuffer(
            replay_buffer, demo_buffer, demo_ratio=args.demo_ratio
        )
        print(
            f"Loaded {n_demos} demo transitions from {demo_dataset.num_episodes} "
            f"episodes; demo_ratio={args.demo_ratio:.2f}"
        )
        if args.learning_starts > 5_000:
            print(
                "Tip: with demonstrations you can lower --learning-starts "
                "(e.g. 1000) to start updating sooner."
            )
        if not args.auto_entropy_tuning:
            print(
                "WARNING: auto entropy tuning is OFF (fixed alpha). On CarRacing "
                "this can collapse to slow/repetitive driving; the previous good "
                "run used --auto-entropy-tuning. Strongly recommended."
            )

    # Initialize action visualizer if requested
    visualizer = None
    if args.visualize_actions:
        visualizer = ActionVisualizer(action_dim, args.env, history_len=100)
        print(f"Action visualization enabled for {args.env}")

    csv_path = run_dir / "training_rewards.csv"
    checkpoint_path = run_dir / "latest.pth"
    best_path = run_dir / "best.pth"
    replay_buffer_path = run_dir / "replay_buffer.pth"

    # Track training state for resume
    start_step = 0
    episode_count = 0
    best_eval = -float("inf")
    csv_mode = "w"  # write mode by default

    # Resume from checkpoint if requested
    if args.resume is not None:
        resume_path = Path(args.resume)
        print(f"Resuming from: {resume_path}")
        checkpoint = agent.load_checkpoint(resume_path)
        start_step = checkpoint.get("global_step", 0)
        best_eval = checkpoint.get("extra", {}).get("best_eval", -float("inf"))
        episode_count = checkpoint.get("extra", {}).get("episode_count", 0)
        
        # Try to load replay buffer from the same directory as the checkpoint
        resume_replay_path = resume_path.parent / "replay_buffer.pth"
        if resume_replay_path.exists():
            replay_buffer.load(str(resume_replay_path))
        else:
            print(f"Warning: No replay buffer found at {resume_replay_path}, starting with empty buffer.")
        
        csv_mode = "a"  # append to existing CSV
        print(f"Resuming from step {start_step}, episode {episode_count}, best_eval={best_eval:.2f}")

        # Adjust total steps: if resuming from step 100k with --steps 200k, train 100k more steps
        if start_step >= args.steps:
            # If already past the target, extend by the requested steps
            print(f"Already past step {args.steps}, extending training by {args.steps} more steps.")
            args.steps = args.steps + start_step

    # Warm-start the actor from a behavior-cloning checkpoint.
    if args.bc_checkpoint is not None:
        bc_path = Path(args.bc_checkpoint)
        if not bc_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {bc_path}")
        bc_checkpoint = torch.load(bc_path, map_location=device, weights_only=False)
        saved_state_dim = bc_checkpoint.get("state_dim")
        if saved_state_dim is not None and tuple(saved_state_dim) != tuple(state_dim):
            raise ValueError(
                f"BC checkpoint state_dim {saved_state_dim} does not match "
                f"environment observation space {state_dim}."
            )
        if not bc_checkpoint.get("use_cnn", use_cnn):
            raise ValueError("BC checkpoint was not trained with the CNN actor.")
        agent.actor.load_state_dict(bc_checkpoint["actor"])
        bc_meta = bc_checkpoint.get("bc", {})
        bc_metric = bc_meta.get("val_action_mse", bc_meta.get("val_loss_pre_tanh"))
        bc_txt = "nan" if bc_metric is None else f"{bc_metric:.4f}"
        print(f"Warm-started actor from {bc_path} (BC val action-MSE={bc_txt})")
        eval_returns = bc_meta.get("eval_returns")
        if eval_returns:
            mean_bc_return = float(np.mean(eval_returns))
            print(f"BC actor rollout mean return: {mean_bc_return:.1f}")
            if mean_bc_return < 100:
                print(
                    "WARNING: this BC actor drives poorly. Warm-starting SAC from it "
                    "can trap training in a slow/random local optimum. Prefer "
                    "recording DAgger corrections, or skip --bc-checkpoint and rely "
                    "on --demo-dir mixing instead. Re-run with --auto-entropy-tuning."
                )

    state, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_length = 0
    episode_start_time = None

    with csv_path.open(csv_mode, newline="") as file:
        writer = csv.writer(file)
        # Only write CSV header when starting fresh (not resuming)
        if csv_mode == "w":
            writer.writerow([
                "step", "episode", "episode_return", "episode_length", "episode_time",
                "eval_mean", "eval_std"
            ])

        episode_start_time = None
        for step in range(start_step + 1, args.steps + 1):
            if step == start_step + 1:
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
                for _ in range(gradient_steps):
                    agent.update(replay_buffer, batch_size)

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

                # Save latest checkpoint with replay buffer
                _save_checkpoint_with_buffer(
                    agent, replay_buffer, checkpoint_path, replay_buffer_path,
                    step, args, state_dim, action_dim, use_cnn,
                    action_low, action_high, best_eval, episode_count,
                )

                if mean_ret > best_eval:
                    best_eval = mean_ret
                    _save_checkpoint_with_buffer(
                        agent, replay_buffer, best_path, replay_buffer_path,
                        step, args, state_dim, action_dim, use_cnn,
                        action_low, action_high, best_eval, episode_count,
                    )

            # Periodic checkpoint saving (every --checkpoint-freq steps)
            if args.checkpoint_freq > 0 and step % args.checkpoint_freq == 0:
                periodic_path = run_dir / f"checkpoint_{step // 1000}k.pth"
                periodic_replay_path = run_dir / f"checkpoint_{step // 1000}k_replay.pth"
                print(f"[checkpoint] Saving periodic checkpoint at step {step}")
                _save_checkpoint_with_buffer(
                    agent, replay_buffer, periodic_path, periodic_replay_path,
                    step, args, state_dim, action_dim, use_cnn,
                    action_low, action_high, best_eval, episode_count,
                )

    # Always save at the end, even if the final step isn't an eval step.
    _save_checkpoint_with_buffer(
        agent, replay_buffer, checkpoint_path, replay_buffer_path,
        args.steps, args, state_dim, action_dim, use_cnn,
        action_low, action_high, best_eval, episode_count,
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
