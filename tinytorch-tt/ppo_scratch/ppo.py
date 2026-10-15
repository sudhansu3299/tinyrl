#ppo algo
#calculates losses, normalizes advantages, surrogate objective, clipping


import torch
import torch.nn.functional as F


def normalize_advantages(advantages, eps=1e-8):
    """Normalize advantages across the complete rollout."""
    return (
            advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + eps)


def compute_ppo_loss(
        model,
        states,
        actions,
        old_log_probs,
        advantages,
        returns,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
):
    # Evaluate the same state-action pairs under the current policy.
    _, new_log_probs, entropy, new_values = (
        model.get_action_and_value(states, action=actions)
    )

    # Importance ratio: new policy / old policy.
    ratios = torch.exp(new_log_probs - old_log_probs)

    # Original and clipped surrogate objectives.
    surrogate1 = ratios * advantages
    clipped_ratios = torch.clamp(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
        )
    surrogate2 = clipped_ratios * advantages

    # Negate because optimizers minimize losses.
    policy_loss = -torch.minimum(
        surrogate1, surrogate2
    ).mean()

    # Critic regression loss.
    value_loss = F.mse_loss(new_values, returns)

    # Mean policy entropy.
    entropy_mean = entropy.mean()

    # Combined objective to minimize.
    total_loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy_mean
    )

    # Useful diagnostic: fraction of ratios outside clip range.
    clip_fraction = (
            (ratios - 1.0).abs() > clip_epsilon
    ).float().mean()

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "clip_fraction": clip_fraction,
        "approx_kl": (old_log_probs - new_log_probs).mean(),
        "ratios": ratios,
    }


def iterate_minibatches(data, advantages, returns,
                        minibatch_size=64, epochs=10):
    """Yield shuffled minibatches for multiple PPO epochs."""
    n = len(data["rewards"])

    for _ in range(epochs):
        indices = torch.randperm(n)

        for start in range(0, n, minibatch_size):
            batch_indices = indices[start:start + minibatch_size]

            yield {
                "states": data["states"][batch_indices],
                "actions": data["actions"][batch_indices],
                "old_log_probs": data["log_probs"][batch_indices],
                "advantages": advantages[batch_indices],
                "returns": returns[batch_indices],
            }

def update_ppo(
        model,
        optimizer,
        data,
        advantages,
        returns,
        minibatch_size=64,
        epochs=10,
        clip_epsilon=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5,
):
    # Normalize advantages once across the complete rollout.
    advantages = normalize_advantages(advantages)

    metrics = {
        "loss": [],
        "policy_loss": [],
        "value_loss": [],
        "entropy": [],
        "clip_fraction": [],
        "approx_kl": [],
        "grad_norm": [],
    }

    for batch in iterate_minibatches(
            data, advantages, returns,
            minibatch_size=minibatch_size,
            epochs=epochs,
    ):
        # 1. Compute the current PPO objective.
        result = compute_ppo_loss(
            model,
            states=batch["states"],
            actions=batch["actions"],
            old_log_probs=batch["old_log_probs"],
            advantages=batch["advantages"],
            returns=batch["returns"],
            clip_epsilon=clip_epsilon,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
        )

        # 2. Clear gradients from the previous minibatch.
        optimizer.zero_grad(set_to_none=True)

        # 3. Compute gradients through both networks.
        result["loss"].backward()

        # 4. Limit excessively large gradient norms.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )

        # 5. Update the network parameters.
        optimizer.step()

        # Save metrics for monitoring.
        for key in metrics:
            if key == "grad_norm":
                metrics[key].append(grad_norm.item())
            else:
                metrics[key].append(result[key].detach().item())

    # Average metrics across all minibatch updates.
    return {
        key: sum(values) / len(values)
        for key, values in metrics.items()
    }