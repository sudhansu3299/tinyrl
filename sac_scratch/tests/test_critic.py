
import torch
import gymnasium as gym

from actor import Actor
from critic import Critic

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = Actor(state_dim, action_dim, action_scale=2.0)

critic1 = Critic(state_dim, action_dim)
critic2 = Critic(state_dim, action_dim)

state, _ = env.reset(seed=42)
state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

action, log_prob = actor.sample(state)

q1 = critic1(state, action)
q2 = critic2(state, action)

print("Action:", action)
print("Q1:", q1)
print("Q2:", q2)

assert q1.shape == (1, 1)
assert q2.shape == (1, 1)

# Verify that gradients can flow through the critic
loss = q1.mean()
loss.backward()

print("Critic test passed!")
env.close()
