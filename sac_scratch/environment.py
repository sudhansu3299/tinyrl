
import gymnasium as gym
import torch

env = gym.make("Pendulum-v1")

obs, info = env.reset(seed=42)

print("PyTorch:", torch.__version__)
print("Observation:", obs)
print("Observation shape:", env.observation_space.shape)
print("Action shape:", env.action_space.shape)
print("Action bounds:", env.action_space.low, env.action_space.high)

action = env.action_space.sample()
next_obs, reward, terminated, truncated, info = env.step(action)

print("Action:", action)
print("Reward:", reward)
print("Next observation:", next_obs)
print("Episode ended:", terminated or truncated)

env.close()
