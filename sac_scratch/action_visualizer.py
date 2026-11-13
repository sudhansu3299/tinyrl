"""
Real-time action visualization for SAC training.
Displays current action values and time-series history in a separate window.
"""
import numpy as np
import matplotlib
# Force a GUI-compatible backend before importing pyplot
_gui_backends = ['TkAgg', 'Qt5Agg', 'GTK3Agg', 'MacOSX']
for _backend in _gui_backends:
    try:
        matplotlib.use(_backend, force=True)
        break
    except Exception:
        continue
else:
    pass

import matplotlib.pyplot as plt
from collections import deque


class ActionVisualizer:
    def __init__(self, action_dim, env_id, history_len=100):
        """
        Initialize action visualizer.

        Args:
            action_dim: Number of action dimensions
            env_id: Environment name for action label mapping
            history_len: Number of timesteps to show in history graph
        """
        self.action_dim = action_dim
        self.history_len = history_len
        self._env_id = env_id
        self.action_names = self._get_action_names(env_id, action_dim)

        # Initialize history buffers
        self.history = [deque([0.0] * history_len, maxlen=history_len) for _ in range(action_dim)]

        # Try to actually create a figure to verify GUI is available
        actual_backend = matplotlib.get_backend()
        gui_available = False
        try:
            _test_fig, _ = plt.subplots(figsize=(1, 1))
            plt.close(_test_fig)
            gui_available = True
        except Exception:
            gui_available = False

        if not gui_available:
            print(f"\n[ActionVisualizer] WARNING: Cannot show visualization window.")
            print(f"  Matplotlib backend '{actual_backend}' is available but no GUI toolkit is installed.")
            print("  To fix this, install one of:")
            print("    sudo apt install python3-tk          # for TkAgg (Ubuntu/Debian)")
            print("    pip install PyQt5                    # for Qt5Agg")
            print("  Then re-run with --visualize-actions.")
            print()
            self.fig = None
        else:
            # Setup figure on the main thread to avoid segfault with pygame
            try:
                self._setup_figure()
                print(f"[ActionVisualizer] Window opened (backend: {actual_backend})")
            except Exception as e:
                print(f"\n[ActionVisualizer] WARNING: Failed to create visualization window: {e}")
                print("  Make sure you have a display server running (X11/Wayland on Linux, or macOS GUI).")
                print()
                self.fig = None

        self.timestep = 0
    
    def _setup_figure(self):
        """Create the matplotlib figure on the main thread."""
        self.fig, (self.ax_bars, self.ax_graph) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle(f"Action Visualization - {self._get_env_display_name()}", fontsize=14, fontweight='bold')
        
        # Setup bar chart (current values)
        self.ax_bars.set_title("Current Action Values", fontsize=12)
        self.ax_bars.set_xlabel("Action")
        self.ax_bars.set_ylabel("Value")
        self.ax_bars.set_ylim(-1.1, 1.1)
        self.bars = self.ax_bars.bar(
            range(self.action_dim), 
            [0.0] * self.action_dim, 
            color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD'][:self.action_dim]
        )
        self.ax_bars.set_xticks(range(self.action_dim))
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
        self.ax_graph.set_title("Action History (Last 100 Timesteps)", fontsize=12)
        self.ax_graph.set_xlabel("Timestep")
        self.ax_graph.set_ylabel("Action Value")
        self.ax_graph.set_ylim(-1.1, 1.1)
        self.ax_graph.grid(True, alpha=0.3)
        
        # Initialize lines for each action dimension
        self.lines = []
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        for i in range(self.action_dim):
            line, = self.ax_graph.plot(
                range(self.history_len), 
                [0.0] * self.history_len,
                label=self.action_names[i],
                color=colors[i % len(colors)],
                linewidth=2
            )
            self.lines.append(line)
        
        self.ax_graph.legend(loc='upper right', fontsize=9)
        
        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.show(block=False)
    
    def _get_env_display_name(self):
        """Short display name for the environment."""
        return self._env_id
    
    def update(self, action):
        """
        Update the visualization with a new action (called on main thread).
        
        Args:
            action: numpy array of shape (action_dim,)
        """
        if self.fig is None:
            return
        
        if isinstance(action, np.ndarray):
            action = action.flatten()
        
        # Update history
        for i in range(self.action_dim):
            self.history[i].append(float(action[i]))
        
        # Update bar chart
        for i, (bar, label) in enumerate(zip(self.bars, self.bar_labels)):
            val = float(action[i])
            bar.set_height(val)
            label.set_text(f'{val:.2f}')
            if val >= 0:
                label.set_y(bar.get_height() + 0.02)
                label.set_va('bottom')
            else:
                label.set_y(bar.get_height() - 0.05)
                label.set_va('top')
        
        # Update time-series graph
        for i, line in enumerate(self.lines):
            line.set_ydata(list(self.history[i]))
        
        self.timestep += 1
        
        # Redraw
        try:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
        except Exception:
            pass  # Window may have been closed
    
    def _get_action_names(self, env_id, action_dim):
        """Get human-readable action names based on environment."""
        env_id_lower = env_id.lower()
        
        if "carracing" in env_id_lower:
            if action_dim == 3:
                return ["Steering", "Gas", "Brake"]
            elif action_dim == 4:
                return ["Steering", "Gas", "Brake", "N/A"]
        elif "bipedalwalker" in env_id_lower:
            if action_dim == 4:
                return ["Hip L", "Knee L", "Hip R", "Knee R"]
        elif "lunarlander" in env_id_lower:
            if action_dim == 2:
                return ["Main Engine", "Side Engine"]
            elif action_dim == 3:
                return ["Main", "Side L", "Side R"]
        elif "halfcheetah" in env_id_lower:
            return [f"Joint {i}" for i in range(action_dim)]
        elif "hopper" in env_id_lower:
            return [f"Joint {i}" for i in range(action_dim)]
        elif "walker2d" in env_id_lower:
            return [f"Joint {i}" for i in range(action_dim)]
        elif "ant" in env_id_lower:
            return [f"Joint {i}" for i in range(action_dim)]
        elif "humanoid" in env_id_lower:
            return [f"Joint {i}" for i in range(action_dim)]
        
        # Fallback to generic names
        return [f"Action {i}" for i in range(action_dim)]
    
    def close(self):
        """Close the visualization window."""
        if getattr(self, 'fig', None) is not None:
            try:
                plt.close(self.fig)
            except Exception:
                pass
