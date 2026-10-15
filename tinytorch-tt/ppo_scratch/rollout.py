
import torch

from buffer import RolloutBuffer


def collect_rollout(
        model,
        env,
        num_steps=1024,
        seed=None,
        observation=None,
        return_observation=False,
):
    buffer = RolloutBuffer()

    # Reset only when starting a fresh training run or evaluation.
    if observation is None:
        observation, _ = env.reset(seed=seed)

    for _ in range(num_steps):
        state = torch.as_tensor(
            observation, dtype=torch.float32
        )

        with torch.no_grad():
            action, log_prob, _, value = (
                model.get_action_and_value(state)
            )

        next_observation, reward, terminated, truncated, _ = (
            env.step(action.item())
        )

        next_state = torch.as_tensor(
            next_observation, dtype=torch.float32
        )

        # Bootstrap unless the environment truly terminated.
        with torch.no_grad():
            next_value = (
                torch.tensor(0.0)
                if terminated
                else model.critic(next_state)
            )

        buffer.add(
            state=state,
            action=action.item(),
            reward=float(reward),
            log_prob=log_prob,
            value=value,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            observation, _ = env.reset()
        else:
            observation = next_observation

    if return_observation:
        return buffer, observation

    return buffer
