#This is to visualize the actions without training and doing random sampled actions in the env.

import argparse
import time

import gymnasium as gym
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Visualize random actions in any Gymnasium environment"
    )
    parser.add_argument(
        "--env",
        required=True,
        default="LunarLanderContinuous-v3",
        help="Gymnasium environment ID, e.g. BipedalWalker-v3 or LunarLanderContinuous-v3",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Log every Nth action (default: every action)",
    )
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be >= 1")
    if args.log_every < 1:
        parser.error("--log-every must be >= 1")

    env = gym.make(args.env, render_mode="human")

    # Seed the action space for reproducible random actions.
    env.action_space.seed(args.seed)

    episode_returns = []
    episode_lengths = []

    try:
        print(f"Environment: {args.env}")
        print("Policy:      Random actions (no trained weights)")
        print(f"Episodes:    {args.episodes}")
        print("Close the simulation window to stop viewing.")

        for episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed + episode)

            total_reward = 0.0
            steps = 0
            terminated = False
            truncated = False

            print(f"\n{'=' * 60}")
            print(f"Starting episode {episode + 1}")
            print(f"Observation shape: {state.shape}")
            print(f"Action space: {env.action_space}")
            print(f"{'=' * 60}")

            while not (terminated or truncated):
                # Sample a random action directly from the environment.
                action = env.action_space.sample()

                next_state, reward, terminated, truncated, info = (
                    env.step(action)
                )

                total_reward += float(reward)
                steps += 1

                # Log the actual action sent to the environment.
                if steps % args.log_every == 0:
                    print(
                        f"Episode {episode + 1} | "
                        f"Step {steps:4d} | "
                        f"Action: {np.array2string(action, precision=3)} | "
                        f"Reward: {reward:7.3f} | "
                        f"Terminated: {terminated} | "
                        f"Truncated: {truncated}"
                    )

                state = next_state

                if args.delay > 0:
                    time.sleep(args.delay)

            episode_returns.append(total_reward)
            episode_lengths.append(steps)

            if terminated and truncated:
                ending = "terminated and truncated"
            elif terminated:
                ending = "terminated by environment"
            elif truncated:
                ending = "truncated (time limit or cutoff)"
            else:
                ending = "unknown"

            print(f"\nEpisode {episode + 1} finished")
            print(f"  Return:     {total_reward:.2f}")
            print(f"  Steps:      {steps}")
            print(f"  Ending:     {ending}")
            print(f"  Terminated: {terminated}")
            print(f"  Truncated:  {truncated}")

    finally:
        env.close()

    print("\n" + "=" * 60)
    print("RANDOM POLICY SUMMARY")
    print("=" * 60)
    print(f"Episodes:     {len(episode_returns)}")
    print(f"Mean return:  {np.mean(episode_returns):.2f}")
    print(f"Std return:   {np.std(episode_returns):.2f}")
    print(f"Mean length:  {np.mean(episode_lengths):.1f} steps")


if __name__ == "__main__":
    main()
