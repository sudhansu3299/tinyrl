
import torch
import gymnasium as gym

from replay_buffer import ReplayBuffer

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

buffer = ReplayBuffer(state_dim, action_dim, capacity=1000)

state, _ = env.reset(seed=42)

# Collect 100 random transitions
for _ in range(100):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated

    buffer.add(state, action, reward, next_state, done)

    state = next_state

    if done:
        state, _ = env.reset()

print("Buffer size:", buffer.size)

states, actions, rewards, next_states, dones = buffer.sample(32)

print("States:", states.shape)
print("Actions:", actions.shape)
print("Rewards:", rewards.shape)
print("Next states:", next_states.shape)
print("Dones:", dones.shape)

assert states.shape == (32, state_dim)
assert actions.shape == (32, action_dim)
assert rewards.shape == (32, 1)

print("Replay buffer test passed!")
env.close()
