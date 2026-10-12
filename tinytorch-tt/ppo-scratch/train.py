
from collections import deque

import torch
import csv
import gymnasium as gym

from models import ActorCritic
from rollout import collect_rollout
from advantages import calculate_gae
from ppo import update_ppo

def train():
    torch.manual_seed(42)

    env = gym.make("CartPole-v1")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(state_dim, action_dim)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=3e-4,
    )

    # PPO hyperparameters
    total_updates = 100
    rollout_steps = 1024
    minibatch_size = 64
    epochs = 10

    recent_rewards = deque(maxlen=20)
    global_step = 0

    with open("training_rewards.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["environment_steps", "episode_reward"])

    for update in range(1, total_updates + 1):
        # Collect fresh experience using the current policy.
        buffer = collect_rollout(
            model,
            env,
            num_steps=rollout_steps,
            seed=42 + update,
        )

        data = buffer.get_tensors()
        global_step += len(buffer)

        # Calculate TD errors, GAE and value targets.
        _, advantages, returns = calculate_gae(data)

        # Update the actor and critic.
        metrics = update_ppo(
            model=model,
            optimizer=optimizer,
            data=data,
            advantages=advantages,
            returns=returns,
            minibatch_size=minibatch_size,
            epochs=epochs,
        )

        # Extract completed episode returns from this rollout.
        completed_returns = []
        episode_return = 0.0

        for i, (reward, terminated, truncated) in enumerate(
                zip(data["rewards"], data["terminated"], data["truncated"])
        ):
            episode_return += reward.item()

            if terminated.item() or truncated.item():
                completed_returns.append(episode_return)
                recent_rewards.append(episode_return)

                episode_step = global_step - len(data["rewards"]) + i + 1

                with open("training_rewards.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([episode_step, episode_return])

                episode_return = 0.0

        if update % 1 == 0:
            mean_reward = (
                sum(recent_rewards) / len(recent_rewards)
                if recent_rewards else float("nan")
            )

            print(
                f"Update {update:03d} | "
                f"Steps {global_step:06d} | "
                f"Mean episode reward (last 20): "
                f"{mean_reward:.2f} | "
                f"Policy loss: {metrics['policy_loss']:.3f} | "
                f"Value loss: {metrics['value_loss']:.3f} | "
                f"Entropy: {metrics['entropy']:.3f} | "
                f"Approx KL: {metrics['approx_kl']:.4f}"
            )

    env.close()

    # Save the trained model.
    torch.save(model.state_dict(), "ppo_cartpole.pth")
    print("Saved model to ppo_cartpole.pth")


if __name__ == "__main__":
    train()
