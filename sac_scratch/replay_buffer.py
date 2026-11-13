
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim, action_dim, capacity=1_000_000, image_obs=False):
        self.capacity = capacity
        self.image_obs = image_obs

        if image_obs:
            # For image observations, use lists to avoid pre-allocating huge arrays
            # (e.g. 1M x 96 x 96 x 3 x 4 bytes ≈ 103 GiB)
            self.states = [None] * capacity
            self.next_states = [None] * capacity
            # Store shape for efficient stacking
            self.state_shape = state_dim
        else:
            self.states = np.zeros((capacity, state_dim), dtype=np.float32)
            self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32) #episode done or not

        self.ptr = 0 # index where next experience will be written to, different when buffer overflows
        self.size = 0 # number of valid experiences stored

    def save(self, path):
        """Save replay buffer contents to disk using torch.save (handles lists and arrays)."""
        data = {
            "states": self.states[:self.size],
            "next_states": self.next_states[:self.size],
            "actions": self.actions[:self.size],
            "rewards": self.rewards[:self.size],
            "dones": self.dones[:self.size],
            "ptr": self.ptr,
            "size": self.size,
        }
        torch.save(data, path)
        print(f"Replay buffer saved: {self.size} transitions -> {path}")

    def load(self, path):
        """Load replay buffer contents from disk."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        loaded_size = data["size"]
        
        # Restore items up to loaded_size
        for i in range(loaded_size):
            self.states[i] = data["states"][i]
            self.next_states[i] = data["next_states"][i]
        self.actions[:loaded_size] = data["actions"]
        self.rewards[:loaded_size] = data["rewards"]
        self.dones[:loaded_size] = data["dones"]
        
        self.ptr = data["ptr"]
        self.size = loaded_size
        print(f"Replay buffer loaded: {self.size} transitions from {path}")

    def add(self, state, action, reward, next_state, done):
        if self.image_obs:
            self.states[self.ptr] = state
            self.next_states[self.ptr] = next_state
        else:
            self.states[self.ptr] = state
            self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        if self.image_obs:
            # Pre-allocate output arrays for faster stacking than np.stack + list comprehension
            states = np.empty((batch_size, *self.state_shape), dtype=np.float32)
            next_states = np.empty((batch_size, *self.state_shape), dtype=np.float32)
            for i, idx in enumerate(indices):
                states[i] = self.states[idx]
                next_states[i] = self.next_states[idx]
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]

        return (
            states,
            self.actions[indices],
            self.rewards[indices],
            next_states,
            self.dones[indices],
        )


class MixedReplayBuffer:
    """A ReplayBuffer that reserves part of every batch for demonstrations.

    New online transitions go to ``online``; ``demo`` is filled once via
    ``demo_utils.fill_buffer_from_demos`` and sampled from at a fixed ratio.
    This is the "keep demos forever" variant of demo-augmented SAC: it avoids
    the BC prior being washed out early in training.

    Saving/loading only touches the online buffer -- demos are reloaded from
    their directory when training is resumed.
    """

    def __init__(self, online, demo=None, demo_ratio=0.3):
        if demo_ratio < 0.0 or demo_ratio > 1.0:
            raise ValueError("demo_ratio must be in [0, 1]")
        self.online = online
        self.demo = demo
        self.demo_ratio = demo_ratio

    @property
    def size(self):
        return self.online.size

    @property
    def capacity(self):
        return self.online.capacity

    @property
    def demo_size(self):
        return 0 if self.demo is None else self.demo.size

    def add(self, *args, **kwargs):
        return self.online.add(*args, **kwargs)

    def sample(self, batch_size):
        if self.demo is None or self.demo.size == 0 or self.demo_ratio <= 0.0:
            return self.online.sample(batch_size)

        n_demo = min(int(round(batch_size * self.demo_ratio)), self.demo.size)
        n_online = min(batch_size - n_demo, self.online.size)
        # If one source is short, backfill from the other so the batch is full.
        n_demo = min(batch_size - n_online, self.demo.size)

        parts = []
        if n_demo > 0:
            parts.append(self.demo.sample(n_demo))
        if n_online > 0:
            parts.append(self.online.sample(n_online))
        if not parts:
            raise RuntimeError("MixedReplayBuffer has no data to sample from")

        return tuple(
            np.concatenate([part[i] for part in parts], axis=0)
            for i in range(5)
        )

    def save(self, path):
        self.online.save(path)

    def load(self, path):
        self.online.load(path)
