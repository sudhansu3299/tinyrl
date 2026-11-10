
import torch
from advantages import calculate_gae

data = {
    "rewards": torch.tensor([1.0, 1.0]),
    "values": torch.tensor([0.5, 0.6]),
    "next_values": torch.tensor([0.6, 0.0]),
    "terminated": torch.tensor([False, True]),
    "truncated": torch.tensor([False, False]),
}

deltas, advantages, returns = calculate_gae(data)

print("TD errors:", deltas)
print("Advantages:", advantages)
print("Returns:", returns)

expected_deltas = torch.tensor([1.094, 0.4])
expected_advantages = torch.tensor([1.4702, 0.4])
expected_returns = torch.tensor([1.9702, 1.0])

torch.testing.assert_close(
    deltas, expected_deltas, atol=1e-5, rtol=1e-5
)
torch.testing.assert_close(
    advantages, expected_advantages, atol=1e-5, rtol=1e-5
)
torch.testing.assert_close(
    returns, expected_returns, atol=1e-5, rtol=1e-5
)

print("GAE tests passed!")
