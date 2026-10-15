
import gymnasium as gym

from sac import SAC
from replay_buffer import ReplayBuffer

env = gym.make("Pendulum-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_scale = float(env.action_space.high[0])

agent = SAC(
    state_dim=state_dim,
    action_dim=action_dim,
    action_scale=action_scale,
)

buffer = ReplayBuffer(
    state_dim=state_dim,
    action_dim=action_dim,
    capacity=10_000,
)

# Collect random transitions to initialize the replay buffer.
state, _ = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()

    next_state, reward, terminated, truncated, _ = env.step(action)

    # Store true termination separately from time-limit truncation.
    buffer.add(
        state, action, reward, next_state, float(terminated)
    )

    state = next_state

    if terminated or truncated:
        state, _ = env.reset()

print("Replay buffer initialized:", buffer.size)

# Run a few SAC gradient updates.
for step in range(5):
    losses = agent.update(buffer, batch_size=256)
    print(f"Update {step + 1}: {losses}")

# Verify action selection.
state, _ = env.reset(seed=123)
action = agent.select_action(state)

print("Selected action:", action)

assert action.shape == (action_dim,)
assert -2.0 <= action[0] <= 2.0

print("SAC update test passed!")

env.close()
