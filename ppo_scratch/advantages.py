#Implement TD errors, GAE
import torch


def calculate_gae(data, gamma=0.99, gae_lambda=0.95):
    rewards = data["rewards"]
    values = data["values"]
    next_values = data["next_values"]
    terminated = data["terminated"]
    truncated = data["truncated"]

    # 1. Compute one-step TD errors.
    # True termination means there is no future value.
    bootstrap_mask = (~terminated).float()

    deltas = (
            rewards
            + gamma * bootstrap_mask * next_values
            - values
    )

    # 2. Compute GAE backwards through the rollout.
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0, dtype=rewards.dtype)

    for t in reversed(range(len(rewards))):
        # doesn't propagate GAE across an episode boundary.
        episode_mask = (~(terminated[t] | truncated[t])).float()

        gae = (
                deltas[t]
                + gamma * gae_lambda * episode_mask * gae
        )

        advantages[t] = gae

    # 3. Critic targets estimated return. and not using mc returns(Gt)
    # Rt = At(GAE) + V(St)
    returns = advantages + values #targeted returns

    return deltas, advantages, returns
