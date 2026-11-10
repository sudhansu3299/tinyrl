
import gymnasium as gym
import torch
from actor import Actor

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = Actor(
    state_dim=state_dim,
    action_dim=action_dim,
    action_scale=float(env.action_space.high[0]),
)

state, _ = env.reset(seed=42)
state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

mean, std = actor(state_tensor)
action, log_prob = actor.sample(state_tensor)

print("Mean:", mean)
print("Std:", std)
print("Sampled action:", action)
print("Log probability:", log_prob)

assert action.shape == (1, action_dim)
assert log_prob.shape == (1, 1)
assert torch.all(action >= -2.0)
assert torch.all(action <= 2.0)

print("Actor test passed!")

env.close()
