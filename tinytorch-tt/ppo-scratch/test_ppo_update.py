
import torch
import gymnasium as gym

from models import ActorCritic
from rollout import collect_rollout
from advantages import calculate_gae
from ppo import update_ppo

torch.manual_seed(42)

env = gym.make("CartPole-v1")

model = ActorCritic(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)

optimizer = torch.optim.Adam(
    model.parameters(), lr=3e-4
)

# Collect experience with the initial policy.
buffer = collect_rollout(
    model, env, num_steps=1024, seed=42
)
data = buffer.get_tensors()

# Calculate TD errors, GAE advantages and returns.
deltas, advantages, returns = calculate_gae(data)

# Save parameters before training.
before = [
    parameter.detach().clone()
    for parameter in model.parameters()
]

# Perform PPO optimization.
metrics = update_ppo(
    model=model,
    optimizer=optimizer,
    data=data,
    advantages=advantages,
    returns=returns,
    minibatch_size=64,
    epochs=10,
)

# Print training diagnostics.
for name, value in metrics.items():
    print(f"{name}: {value:.6f}")

# Confirm that at least one parameter changed.
changed = any(
    not torch.equal(old, new.detach())
    for old, new in zip(before, model.parameters())
)
assert changed, "Model parameters did not change!"

# Confirm that the returned metrics are finite.
assert all(
    torch.isfinite(torch.tensor(value)).item()
    for value in metrics.values()
)

print("\nPPO update tests passed!")
env.close()
