
import gymnasium as gym

env = gym.make("LunarLanderContinuous-v3", render_mode="human")

obs, info = env.reset(seed=42)

print("Observation shape:", obs.shape)
print("Observation space:", env.observation_space)
print("Action space:", env.action_space)

for step in range(1000):
    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print(f"Episode ended at step {step + 1}")
        obs, info = env.reset()

env.close()
