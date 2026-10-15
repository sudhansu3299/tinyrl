import gymnasium as gym

env = gym.make("BipedalWalker-v3")

print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

obs, info = env.reset(seed=42)
print("Observation shape:", obs.shape)

env.close()