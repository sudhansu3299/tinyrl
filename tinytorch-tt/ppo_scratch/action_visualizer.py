"""
Real-time action visualization for PPO training.
Displays action values and time-series history in a separate window.
"""
import numpy as np
import matplotlib.pyplot as plt
from collections import deque


class ActionVisualizer:
    def __init__(self, action_dim, env_id, history_len=100):
        """
        Initialize action visualizer for discrete action spaces.
        
        Args:
            action_dim: Number of discrete actions
            env_id: Environment name for action label mapping
            history_len: Number of timesteps to show in history graph
        """
        self.action_dim = action_dim
        self.history_len = history_len
        self.action_names = self._get_action_names(env_id, action_dim)
        
        # Initialize history buffers for action values
        self.value_history = deque(maxlen=history_len)
        for _ in range(action_dim):
            self.value_history.append(deque([0.0] * history_len, maxlen=history_len))
        
        # Initialize history buffer for selected actions
        self.action_history = deque([0] * history_len, maxlen=history_len)
        
        # Setup figure
        plt.ion()  # Turn on interactive mode
        self.fig, (self.ax_bars, self.ax_graph) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle(f"Action Visualization - {env_id}", fontsize=14, fontweight='bold')
        
        # Setup bar chart (current action values)
        self.ax_bars.set_title("Current Action Values", fontsize=12)
        self.ax_bars.set_xlabel("Action")
        self.ax_bars.set_ylabel("Value")
        self.ax_bars.set_ylim(-1.1, 1.1)
        self.bars = self.ax_bars.bar(
            range(action_dim), 
            [0.0] * action_dim, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][:action_dim]
        )
        self.ax_bars.set_xticks(range(action_dim))
        self.ax_bars.set_xticklabels(self.action_names, rotation=45, ha='right')
        
        # Add value labels on bars
        self.bar_labels = []
        for i, bar in enumerate(self.bars):
            label = self.ax_bars.text(
                bar.get_x() + bar.get_width() / 2, 
                bar.get_height() + 0.02,
                f'{0.0:.2f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
            self.bar_labels.append(label)
        
        # Setup time-series graph
        self.ax_graph.set_title("Action Value History (Last 100 Timesteps)", fontsize=12)
        self.ax_graph.set_xlabel("Timestep")
        self.ax_graph.set_ylabel("Action Value")
        self.ax_graph.set_ylim(-1.1, 1.1)
        self.ax_graph.grid(True, alpha=0.3)
        
        # Initialize lines for each action dimension
        self.lines = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for i in range(action_dim):
            line, = self.ax_graph.plot(
                range(history_len), 
                [0.0] * history_len,
                label=self.action_names[i],
                color=colors[i % len(colors)],
                linewidth=2
            )
            self.lines.append(line)
        
        self.ax_graph.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        self.timestep = 0
    
    def _get_action_names(self, env_id, action_dim):
        """Get human-readable action names based on environment."""
        env_id_lower = env_id.lower()
        
        if "cartpole" in env_id_lower:
            if action_dim == 2:
                return ["Left", "Right"]
        elif "lunarlander" in env_id_lower:
            if action_dim == 4:
                return ["Do Nothing", "Fire Left", "Fire Main", "Fire Right"]
        elif "acrobot" in env_id_lower:
            if action_dim == 3:
                return ["Torque -1", "Torque 0", "Torque +1"]
        elif "mountaincar" in env_id_lower:
            if action_dim == 3:
                return ["Accelerate Left", "Coast", "Accelerate Right"]
        
        # Fallback to generic names
        return [f"Action {i}" for i in range(action_dim)]
    
    def update(self, action_values, selected_action=None):
        """
        Update visualization with new action values.
        
        Args:
            action_values: numpy array of shape (action_dim,) with action values (logits/Q-values)
            selected_action: int, the selected action index (optional)
        """
        if isinstance(action_values, np.ndarray):
            action_values = action_values.flatten()
        
        # Normalize action values to [-1, 1] range for visualization
        if action_values.max() != action_values.min():
            action_values = 2 * (action_values - action_values.min()) / (action_values.max() - action_values.min()) - 1
        
        # Update value history
        for i in range(self.action_dim):
            self.value_history[i].append(float(action_values[i]))
        
        # Update action history
        if selected_action is not None:
            self.action_history.append(selected_action)
        else:
            # If no selected action provided, use argmax
            self.action_history.append(np.argmax(action_values))
        
        # Update bar chart
        for i, (bar, label) in enumerate(zip(self.bars, self.bar_labels)):
            bar.set_height(float(action_values[i]))
            label.set_text(f'{float(action_values[i]):.2f}')
            label.set_y(bar.get_height() + 0.02 if action_values[i] >= 0 else bar.get_height() - 0.05)
        
        # Highlight selected action bar
        if selected_action is not None:
            for i, bar in enumerate(self.bars):
                if i == selected_action:
                    bar.set_edgecolor('black')
                    bar.set_linewidth(3)
                else:
                    bar.set_edgecolor('none')
                    bar.set_linewidth(0)
        
        # Update time-series graph
        for i, line in enumerate(self.lines):
            line.set_ydata(list(self.value_history[i]))
        
        self.timestep += 1
        
        # Redraw
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def close(self):
        """Close the visualization window."""
        plt.ioff()
        plt.close(self.fig)
