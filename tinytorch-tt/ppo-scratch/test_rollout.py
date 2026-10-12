
import torch
import gymnasium as gym

from models import ActorCritic
from rollout import collect_rollout

torch.manual_seed(42)

env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

model = ActorCritic(state_dim, action_dim)

buffer = collect_rollout(
    model=model,
    env=env,
    num_steps=1024,
    seed=42,
)

data = buffer.get_tensors()

print("Transitions collected:", len(buffer))

for name, tensor in data.items():
    print(f"{name}: {tensor.shape}, {tensor.dtype}")

assert len(buffer) == 1024
assert data["states"].shape == (1024, state_dim)
assert data["actions"].shape == (1024,)
assert data["log_probs"].shape == (1024,)
assert data["values"].shape == (1024,)
assert data["next_values"].shape == (1024,)

# Verify that rollout data is not connected to autograd.
assert not data["log_probs"].requires_grad
assert not data["values"].requires_grad

print("\nRollout tests passed!")

env.close()

from advantages import calculate_gae

# Existing code: collect the rollout and get tensors.
data = buffer.get_tensors()

deltas, advantages, returns = calculate_gae(data)

print("Mean TD error:", deltas.mean().item())
print("Mean advantage:", advantages.mean().item())
print("Mean return target:", returns.mean().item())

assert advantages.shape == (1024,)
assert returns.shape == (1024,)
assert torch.isfinite(advantages).all()
assert torch.isfinite(returns).all()

print("Rollout + GAE tests passed!")
