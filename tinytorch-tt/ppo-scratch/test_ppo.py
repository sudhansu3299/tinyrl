
import torch
import gymnasium as gym

from models import ActorCritic
from rollout import collect_rollout
from advantages import calculate_gae
from ppo import (
    normalize_advantages,
    compute_ppo_loss,
    iterate_minibatches,
)

torch.manual_seed(42)

env = gym.make("CartPole-v1")
model = ActorCritic(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.n,
)

# Collect experience with the current policy.
buffer = collect_rollout(model, env, num_steps=1024, seed=42)
data = buffer.get_tensors()

# Calculate TD errors, advantages and critic targets.
deltas, advantages, returns = calculate_gae(data)
advantages = normalize_advantages(advantages)

# Test a single minibatch.
batch = next(iterate_minibatches(
    data, advantages, returns,
    minibatch_size=64, epochs=1,
))

result = compute_ppo_loss(
    model,
    states=batch["states"],
    actions=batch["actions"],
    old_log_probs=batch["old_log_probs"],
    advantages=batch["advantages"],
    returns=batch["returns"],
)

for key in (
        "loss", "policy_loss", "value_loss",
        "entropy", "clip_fraction", "approx_kl"
):
    print(f"{key}: {result[key].item():.6f}")

# The loss should be finite and differentiable.
assert torch.isfinite(result["loss"])
assert result["loss"].requires_grad
assert result["ratios"].shape == (64,)

# Before any parameter update, ratios should be approximately 1.
torch.testing.assert_close(
    result["ratios"],
    torch.ones_like(result["ratios"]),
    atol=1e-5,
    rtol=1e-5,
)

# Check that all expected minibatches are produced.
batches = list(iterate_minibatches(
    data, advantages, returns,
    minibatch_size=64, epochs=10,
))
assert len(batches) == 160  # 16 minibatches x 10 epochs

print("\nPPO objective tests passed!")
env.close()
