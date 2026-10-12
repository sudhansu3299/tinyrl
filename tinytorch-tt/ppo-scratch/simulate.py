
import time
import torch
import gymnasium as gym

from models import ActorCritic


def simulate(checkpoint="ppo_cartpole.pth", episodes=3):
    env = gym.make("CartPole-v1", render_mode="human")

    model = ActorCritic(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
    )

    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=2000 + episode)
            total_reward = 0
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

                # Slow down playback so you can watch the simulation.
                time.sleep(0.01)

            print(
                f"Episode {episode + 1}: reward = {total_reward}"
            )

    finally:
        env.close()


if __name__ == "__main__":
    simulate()
