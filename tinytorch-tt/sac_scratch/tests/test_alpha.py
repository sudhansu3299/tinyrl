
import gymnasium as gym
import torch

from sac import SAC
from replay_buffer import ReplayBuffer

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

initial_alpha = agent.alpha

for step in range(20):
    metrics = agent.update(buffer, batch_size=256)

    print(
        f"Step {step + 1:02d} | "
        f"Alpha: {metrics['alpha']:.4f} | "
        f"Alpha loss: {metrics['alpha_loss']:.4f}"
    )

assert agent.alpha > 0
assert torch.isfinite(agent.log_alpha).all()
assert agent.alpha != initial_alpha

print(f"\nInitial alpha: {initial_alpha:.4f}")
print(f"Final alpha:   {agent.alpha:.4f}")
print("Automatic entropy tuning test passed!")

env.close()
