import argparse
import time

import gymnasium as gym
import numpy as np
import pygame
import torch

from models import ActorCritic


def simulate(env_id, checkpoint, episodes=5, scale=2):
    env = gym.make(env_id, render_mode="rgb_array")

    try:
        # Infer model dimensions from the environment.
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError("Expected a Box observation space.")

        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError(
                "This ActorCritic supports discrete actions only."
            )

        model = ActorCritic(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
        )

        saved = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        # Support full training checkpoints and weights-only checkpoints.
        state_dict = saved.get("model_state_dict", saved)
        model.load_state_dict(state_dict)
        model.eval()

        pygame.init()

        env.reset(seed=2000)
        frame = env.render()
        frame_height, frame_width = frame.shape[:2]

        window = pygame.display.set_mode(
            (frame_width * scale, frame_height * scale),
            pygame.RESIZABLE,
        )
        pygame.display.set_caption(f"PPO Simulation: {env_id}")

        clock = pygame.time.Clock()

        for episode in range(episodes):
            observation, _ = env.reset(seed=2000 + episode)
            total_reward = 0.0
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return

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

                frame = env.render()
                surface = pygame.surfarray.make_surface(
                    np.transpose(frame, (1, 0, 2))
                )

                surface = pygame.transform.scale(
                    surface, window.get_size()
                )

                window.blit(surface, (0, 0))
                pygame.display.flip()

                clock.tick(60)
                time.sleep(0.005)

            print(
                f"Episode {episode + 1}: "
                f"reward = {total_reward:.1f}"
            )
            time.sleep(1)

    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--scale", type=int, default=2)

    args = parser.parse_args()

    simulate(
        env_id=args.env,
        checkpoint=args.checkpoint,
        episodes=args.episodes,
        scale=args.scale,
    )