"""
Test script to verify CNN feature extraction works with CarRacing-v3.
"""
import gymnasium as gym
import torch
import numpy as np

from sac import SAC
from replay_buffer import ReplayBuffer


def test_carracing():
    print("Testing CarRacing-v3 with CNN feature extraction...")
    
    # Create environment
    env = gym.make("CarRacing-v3")
    print(f"Environment: CarRacing-v3")
    print(f"Observation space shape: {env.observation_space.shape}")
    print(f"Action space shape: {env.action_space.shape}")
    
    # Get initial state
    state, _ = env.reset(seed=42)
    print(f"Initial state shape: {state.shape}")
    print(f"Initial state dtype: {state.dtype}")
    print(f"Initial state range: [{state.min()}, {state.max()}]")
    
    # Create agent with CNN
    state_dim = env.observation_space.shape
    action_dim = int(np.prod(env.action_space.shape))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    agent = SAC(
        state_dim=state_dim,
        action_dim=action_dim,
        action_scale=1.0,
        device=device,
        auto_entropy_tuning=True,
        use_cnn=True,
    )
    print("Agent created successfully with CNN enabled")
    
    # Test forward pass
    print("\nTesting forward pass...")
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"State tensor shape: {state_tensor.shape}")
    
    mean, std = agent.actor(state_tensor)
    print(f"Actor output - mean shape: {mean.shape}, std shape: {std.shape}")
    
    action = agent.select_action(state, evaluate=False)
    print(f"Sampled action shape: {action.shape}")
    print(f"Sampled action: {action}")
    
    # Test replay buffer with image observations
    print("\nTesting replay buffer with image observations...")
    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        capacity=1000,
        image_obs=True,
    )
    
    # Add a few transitions
    for i in range(5):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state
        if done:
            state, _ = env.reset()
    
    print(f"Replay buffer size: {replay_buffer.size}")
    print(f"Replay buffer states shape: {replay_buffer.states.shape}")
    
    # Test sampling from buffer
    print("\nTesting sampling from replay buffer...")
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size=2)
    print(f"Sampled states shape: {states.shape}")
    print(f"Sampled actions shape: {actions.shape}")
    print(f"Sampled rewards shape: {rewards.shape}")
    
    # Test update with sampled batch
    print("\nTesting agent update with sampled batch...")
    metrics = agent.update(replay_buffer, batch_size=2)
    print(f"Update metrics: {metrics}")
    
    print("\n✅ All tests passed!")
    env.close()


if __name__ == "__main__":
    test_carracing()
