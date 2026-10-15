
import argparse
import csv
import os
from collections import deque

import torch
import gymnasium as gym

from models import ActorCritic
from rollout import collect_rollout
from advantages import calculate_gae
from ppo import update_ppo
from checkpoint import save_checkpoint, load_checkpoint


def train(env_id, total_steps, rollout_steps=1024):
    torch.manual_seed(42)

    # Use a separate output directory for each environment.
    output_dir = f"runs/{env_id}"
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(env_id)
    observation, _ = env.reset(seed=42)


    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4
    )

    resume_path = None  # Set this when resuming.

    # if resume_path is not None:
    #     global_step, update = load_checkpoint(
    #         resume_path,
    #         model,
    #         optimizer,
    #     )
    #
    #     print(
    #         f"Resumed from {resume_path}: "
    #         f"{global_step} steps, update {update}"
    #     )

    if resume_path is not None:
        try:
            global_step, update = load_checkpoint(
                resume_path, model, optimizer
            )
        except (KeyError, TypeError):
            # Migrate an older model-only checkpoint.
            old_state = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=True,
            )
            model.load_state_dict(old_state)
            global_step = 100000
            update = global_step // rollout_steps

            print("Loaded legacy model weights; optimizer reset.")


    recent_rewards = deque(maxlen=20)
    episode_return = 0.0
    global_step = 0
    update = 0

    reward_path = os.path.join(
        output_dir, "training_rewards.csv"
    )

    with open(reward_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["environment_steps", "episode_reward"])

    while global_step < total_steps:
        steps_this_rollout = min(
            rollout_steps, total_steps - global_step
        )

        buffer, observation = collect_rollout(
            model=model,
            env=env,
            num_steps=steps_this_rollout,
            observation=observation,
            return_observation=True,
        )

        data = buffer.get_tensors()
        global_step += len(buffer)
        update += 1

        _, advantages, returns = calculate_gae(data)

        metrics = update_ppo(
            model=model,
            optimizer=optimizer,
            data=data,
            advantages=advantages,
            returns=returns,
            minibatch_size=64,
            epochs=10,
        )

        # Preserve the running episode return across rollouts.

        rollout_start_step = global_step - len(buffer)

        with open(reward_path, "a", newline="") as f:
            writer = csv.writer(f)

            for i, (reward, terminated, truncated) in enumerate(
                    zip(
                        data["rewards"],
                        data["terminated"],
                        data["truncated"],
                    )
            ):
                episode_return += reward.item()

                if terminated.item() or truncated.item():
                    recent_rewards.append(episode_return)

                    writer.writerow([
                        rollout_start_step + i + 1,
                        episode_return,
                        ])

                    episode_return = 0.0

        if update % 5 == 0 or global_step >= total_steps:
            mean_reward = (
                sum(recent_rewards) / len(recent_rewards)
                if recent_rewards else float("nan")
            )

            print(
                f"Update {update:04d} | "
                f"Steps {global_step:07d} | "
                f"Mean reward (last 20): {mean_reward:.2f} | "
                f"Policy loss: {metrics['policy_loss']:.3f} | "
                f"Value loss: {metrics['value_loss']:.3f} | "
                f"Entropy: {metrics['entropy']:.3f} | "
                f"Approx KL: {metrics['approx_kl']:.4f}"
            )

        # Save the latest checkpoint.
        save_checkpoint(
            os.path.join(output_dir, "latest.pth"),
            model,
            optimizer,
            global_step,
            update,
        )

    env.close()
    print(f"Saved checkpoint in {output_dir}")
    print(f"Saved rewards to {reward_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="CartPole-v1")
    parser.add_argument("--steps", type=int, default=102400)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    train(args.env, args.steps)
