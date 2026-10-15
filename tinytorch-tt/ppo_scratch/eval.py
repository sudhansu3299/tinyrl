
import argparse
import numpy as np
import torch
import gymnasium as gym

from models import ActorCritic


def evaluate(checkpoint_path, episodes=100):
    env = gym.make("LunarLander-v3")

    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Support both full checkpoints and legacy model-only files.
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    rewards = []

    for episode in range(episodes):
        observation, _ = env.reset(seed=1000 + episode)
        total_reward = 0.0
        done = False

        while not done:
            state = torch.as_tensor(
                observation, dtype=torch.float32
            )

            # Deterministic evaluation and not sampling
            with torch.no_grad():
                logits = model.actor(state)
                action = torch.argmax(logits).item()

            observation, reward, terminated, truncated, _ = (
                env.step(action)
            )

            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

    env.close()

    rewards = np.asarray(rewards)
    print(f"Episodes: {episodes}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Std deviation: {rewards.std():.2f}")
    print(f"Minimum reward: {rewards.min():.2f}")
    print(f"Maximum reward: {rewards.max():.2f}")
    print(f"Episodes with reward >= 200: "
          f"{100 * np.mean(rewards >= 200):.1f}%")

    return rewards


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="runs/LunarLander-v3/latest.pth",
    )
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()

    evaluate(args.checkpoint, args.episodes)
