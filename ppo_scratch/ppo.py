#ppo algo
#calculates losses, normalizes advantages, surrogate objective, clipping


import torch
import torch.nn.functional as F


def normalize_advantages(advantages, eps=1e-8):
    #This helps in stabilize the training process, keeps it more consistent during optimization as At ~ N(0,1)
    return (
            advantages - advantages.mean()
    ) / (advantages.std(unbiased=False) + eps)

    #+ve At: action better then critic expected
    #-ve At: action worse than critic expected


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
    # new policy's log prob, entropy, current critic's value
    _, new_log_probs, entropy, new_values = (
        model.get_action_and_value(states, action=actions)
    )

    # Importance ratio(rt(theta)= new policy / old policy. e^(log(new_policy) - log(old_policy))
    ratios = torch.exp(new_log_probs - old_log_probs)

    # Original and clipped surrogate objectives.
    surrogate1 = ratios * advantages
    clipped_ratios = torch.clamp(
        ratios,
        1.0 - clip_epsilon,
        1.0 + clip_epsilon,
        )
    surrogate2 = clipped_ratios * advantages

    # Negate this because optimizers minimize losses and we maximized it
    #L(pg) = E[min(ratios * A, clip(ratios, 1-epsilon, 1+epsilon) * A)]
    policy_loss = -torch.minimum(
        surrogate1, surrogate2
    ).mean()

    # Critic regression loss.
    #L(vf) = E[(V_phi(st) - Rt)^2]
    value_loss = F.mse_loss(new_values, returns)

    # Mean policy entropy.
    entropy_mean = entropy.mean()

    #  objective is to find the minima of this loss curve.
    total_loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy_mean
    )

    # fraction of ratios outside clip range, just a diagnostic metric
    # high clip fraction means policy gets changed too much during optimization
    clip_fraction = (
            (ratios - 1.0).abs() > clip_epsilon
    ).float().mean()

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "entropy": entropy_mean,
        "clip_fraction": clip_fraction,
        "approx_kl": (old_log_probs - new_log_probs).mean(), #low is better for stable training, less change compared to old
        "ratios": ratios,
    }


def iterate_minibatches(data, advantages, returns,
                        minibatch_size=64, epochs=10):
    # shuffled minibatches for multiple PPO epochs, here (1024/16)*10 = 160 updates
    n = len(data["rewards"])

    for _ in range(epochs):
        indices = torch.randperm(n)

        for start in range(0, n, minibatch_size):
            batch_indices = indices[start : start + minibatch_size]

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
        # Compute the current PPO objective.
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

        # clearing up gradients from the previous minibatch.
        optimizer.zero_grad(set_to_none=True)

        # compute gradients through (Ltotal = Lpg + Lvf - Hentropy) function
        result["loss"].backward()

        # Limit excessively large gradient norms.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm
        )

        # updating the parameters
        optimizer.step()

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