
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


def train(env_name, total_steps, rollout_steps=1024, resume_path=None):
    torch.manual_seed(42) #init seed makes the init weights fixed for reproducibility

    # Use a separate output directory for each env.
    output_dir = f"runs/{env_name}"
    os.makedirs(output_dir, exist_ok=True)

    env = gym.make(env_name)
    observation, _ = env.reset(seed=42)


    model = ActorCritic(
        state_dim=env.observation_space.shape[0], #gym provided 1 D arr
        action_dim=env.action_space.n, # n is for discrete space
    )

    optimizer = torch.optim.Adam(
        model.parameters(), lr=3e-4 #recursively gets all params from both net
    )

    #initializing training checkpoint variables
    global_step = 0 # total interactions completed yet
    update = 0 # total updates as per rollout completed yet

    if resume_path is not None:
        try:
            global_step, update = load_checkpoint(
                resume_path, model, optimizer
            )
            print(
                f"Resumed from {resume_path}: "
                f"{global_step} steps, update {update}"
            )
        except (KeyError, TypeError):
            # legacy checkpoint: model weights only, not optimizer state
            old_state = torch.load(
                resume_path,
                map_location="cpu",
                weights_only=True, #
            )
            model.load_state_dict(old_state) #incase we have only saved model and not optim

            # The original training progress is unknown.
            global_step = 0
            update = 0

            print(
                "Loaded legacy model weights; "
                "training progress and optimizer state unavailable."
            )

    recent_rewards = deque(maxlen=20)
    episode_return = 0.0 #episodic return of the current episode

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
        update += 1 # rollout level update counter

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

        # Preserve the running episode return across rollouts for metrics

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
                episode_return += reward.item() #total reward of the current episode

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

    train(args.env, args.steps, resume_path=args.resume)
