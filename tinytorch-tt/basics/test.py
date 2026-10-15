
import torch
from torch.distributions import Normal

# # Hypothetical outputs from the actor network
# mean = torch.tensor([0.2, -0.5, 0.8, 0.0])
# log_std = torch.tensor([-1.2, -1.6, -0.9, -0.7])
#
# # Convert log standard deviation to standard deviation
# std = log_std.exp()
#
# # Construct a Gaussian distribution
# distribution = Normal(mean, std)
# print("mean:", distribution.loc)
# print("std:", distribution.scale)
#
#
# # Reparametrize and sample a vector
# # raw action: mean + sigma * epsilon
# raw_action = distribution.rsample()
# print("Raw action: ", raw_action)
#
# # Squash the sampled action into (-1, 1)
# action = torch.tanh(raw_action)
#
# print("Environment action:", action)
#
# x = torch.tensor(0.0)
# print(x)

import gymnasium as gym

# 1. Create a classic environment (e.g., CartPole-v1)
env = gym.make("CartPole-v1", render_mode="rgb_array")

# 2. Reset the environment to start a new episode
state, info = env.reset(seed=42)
print("--- Initial Reset ---")
print(f"Initial State: {state}")
print(f"Initial Info:  {info}\n")

# 3. Sample a random valid action from the action space
action = env.action_space.sample()
print(f"Sampled Action: {action}\n")

# 4. Take a step using env.step(action)
next_state, reward, terminated, truncated, info = env.step(action)

print("--- What env.step(action) Returns ---")
print(f"1. next_state  : {next_state} (Type: {type(next_state)})")
print(f"2. reward      : {reward} (Type: {type(reward)})")
print(f"3. terminated  : {terminated} (Type: {type(terminated)})")
print(f"4. truncated   : {truncated} (Type: {type(truncated)})")
print(f"5. info        : {info} (Type: {type(info)})")

env.close()