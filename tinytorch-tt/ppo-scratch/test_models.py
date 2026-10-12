import torch
from models import ActorCritic

torch.manual_seed(42)

state_dim = 4   # CartPole observation dimension
action_dim = 2  # Left or right

model = ActorCritic(state_dim, action_dim)

# Test one state.
state = torch.randn(state_dim)

action, log_prob, entropy, value = (
    model.get_action_and_value(state)
)

print("Single state")
print("Action:", action.item())
print("Log probability:", log_prob.item())
print("Entropy:", entropy.item())
print("Value:", value.item())

assert action.shape == torch.Size([])
assert log_prob.shape == torch.Size([])
assert entropy.shape == torch.Size([])
assert value.shape == torch.Size([])

# Test a batch of 32 states.
states = torch.randn(32, state_dim)

actions, log_probs, entropies, values = (
    model.get_action_and_value(states)
)

print("\nBatch shapes")
print("Actions:", actions.shape)
print("Log probabilities:", log_probs.shape)
print("Entropies:", entropies.shape)
print("Values:", values.shape)

assert actions.shape == (32,)
assert log_probs.shape == (32,)
assert entropies.shape == (32,)
assert values.shape == (32,)

# Test that gradients can flow through both networks.
loss = -log_probs.mean() + values.square().mean()
loss.backward()

assert model.actor.network[-1].weight.grad is not None
assert model.critic.network[-1].weight.grad is not None

print("\nAll tests passed!")
