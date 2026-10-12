
import torch
import gymnasium as gym

from buffer import RolloutBuffer


def collect_rollout(model, env, num_steps=1024, seed=42):
    buffer = RolloutBuffer()

    observation, _ = env.reset(seed=seed)

    for step in range(num_steps):
        # Convert the observation to a PyTorch tensor.
        state = torch.as_tensor(
            observation, dtype=torch.float32
        )

        # Collection does not require gradients.
        with torch.no_grad():
            action, log_prob, _, value = (
                model.get_action_and_value(state)
            )

        # Interact with Gymnasium.
        next_observation, reward, terminated, truncated, _ = (
            env.step(action.item())
        )

        next_state = torch.as_tensor(
            next_observation, dtype=torch.float32
        )

        # Bootstrap the next value except at true termination.
        with torch.no_grad():
            if terminated:
                next_value = torch.tensor(0.0)
            else:
                next_value = model.critic(next_state)

        # Store the transition before resetting the environment.
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

        observation = next_observation

        # Start a new episode when the current one ends.
        if terminated or truncated:
            observation, _ = env.reset()

    return buffer
