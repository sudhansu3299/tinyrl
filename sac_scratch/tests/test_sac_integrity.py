
import copy
import torch
import gymnasium as gym

from sac import SAC
from replay_buffer import ReplayBuffer

torch.manual_seed(42)

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

agent = SAC(state_dim, action_dim, action_scale=2.0)

buffer = ReplayBuffer(state_dim, action_dim, capacity=2000)

state, _ = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, _ = env.step(action)

    buffer.add(
        state, action, reward, next_state, float(terminated)
    )

    state = next_state

    if terminated or truncated:
        state, _ = env.reset()

# 1. Losses must remain finite.
losses = agent.update(buffer, batch_size=256)
assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
print("PASS: all losses are finite")

# 2. Actor parameters must change after an update.
before = [
    p.detach().clone() for p in agent.actor.parameters()
]
agent.update(buffer, batch_size=256)

actor_changed = any(
    not torch.equal(old, new.detach())
    for old, new in zip(before, agent.actor.parameters())
)
assert actor_changed
print("PASS: actor parameters update")

# 3. Target critics must move toward online critics.
target_before = [
    p.detach().clone()
    for p in agent.target_critic1.parameters()
]
online_before = [
    p.detach().clone()
    for p in agent.critic1.parameters()
]

agent.update(buffer, batch_size=256)

target_after = list(agent.target_critic1.parameters())
online_after = list(agent.critic1.parameters())

target_moved = any(
    not torch.equal(old, new.detach())
    for old, new in zip(target_before, target_after)
)
assert target_moved
print("PASS: target critics update")

# 4. Target critics must not receive gradients.
assert all(
    p.grad is None for p in agent.target_critic1.parameters()
)
assert all(
    p.grad is None for p in agent.target_critic2.parameters()
)
print("PASS: target critics have no gradients")

env.close()
print("All integrity checks passed!")
