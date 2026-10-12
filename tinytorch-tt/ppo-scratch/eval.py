
import numpy as np
import torch
import gymnasium as gym

from models import ActorCritic


def evaluate(checkpoint="ppo_cartpole.pth", episodes=100):
    env = gym.make("CartPole-v1")

    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
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

    print(f"Episodes: {episodes}")
    print(f"Mean reward: {np.mean(rewards):.2f}")
    print(f"Std deviation: {np.std(rewards):.2f}")
    print(f"Min reward: {np.min(rewards):.0f}")
    print(f"Max reward: {np.max(rewards):.0f}")
    print(f"Success rate (reward >= 475): "
          f"{100 * np.mean(np.array(rewards) >= 475):.1f}%")

    return rewards


if __name__ == "__main__":
    evaluate()
