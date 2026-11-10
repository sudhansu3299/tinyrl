"""
Test script for action visualizer.
Tests with different environments to ensure proper action labeling and visualization.
"""
import numpy as np
import time
from action_visualizer import ActionVisualizer


def test_visualizer(env_id, action_dim, test_duration=5):
    """Test visualizer with a specific environment configuration."""
    print(f"\nTesting visualizer for {env_id} (action_dim={action_dim})")
    print("=" * 60)
    
    visualizer = ActionVisualizer(action_dim, env_id, history_len=100)
    
    print(f"Action names: {visualizer.action_names}")
    
    # Simulate random actions for test_duration seconds
    start_time = time.time()
    timestep = 0
    while time.time() - start_time < test_duration:
        # Generate random action in [-1, 1] range
        action = np.random.uniform(-1, 1, action_dim)
        visualizer.update(action)
        
        timestep += 1
        time.sleep(0.05)  # 20 Hz update rate
    
    print(f"Updated visualizer {timestep} times over {test_duration} seconds")
    
    visualizer.close()
    print("✅ Test passed!")


def main():
    print("Testing Action Visualizer")
    print("=" * 60)
    
    # Test CarRacing-v3 (3 actions: steering, gas, brake)
    test_visualizer("CarRacing-v3", 3, test_duration=3)
    
    # Test BipedalWalker-v3 (4 actions: hip L, knee L, hip R, knee R)
    test_visualizer("BipedalWalker-v3", 4, test_duration=3)
    
    # Test LunarLanderContinuous-v3 (2 actions: main, side)
    test_visualizer("LunarLanderContinuous-v3", 2, test_duration=3)
    
    # Test generic environment (6 actions)
    test_visualizer("GenericEnv-v0", 6, test_duration=3)
    
    print("\n" + "=" * 60)
    print("All visualizer tests completed successfully!")


if __name__ == "__main__":
    main()
