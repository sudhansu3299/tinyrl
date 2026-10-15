#Store Rollout, Transitions, log probs of trajectories


import torch


class RolloutBuffer: #Stores the transitions carried during one rollout
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.truncated = []

    def add(
            self,
            state,
            action,
            reward,
            log_prob,
            value,
            next_value,
            terminated,
            truncated,
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.next_values.append(next_value)
        self.terminated.append(terminated)
        self.truncated.append(truncated)

    def get_tensors(self):
        """Convert stored transitions into tensors for PPO."""

        return {
            "states": torch.stack(self.states),
            "actions": torch.tensor(self.actions, dtype=torch.long),
            "rewards": torch.tensor(self.rewards, dtype=torch.float32),
            "log_probs": torch.stack(self.log_probs),
            "values": torch.stack(self.values),
            "next_values": torch.stack(self.next_values),
            "terminated": torch.tensor(
                self.terminated, dtype=torch.bool
            ),
            "truncated": torch.tensor(
                self.truncated, dtype=torch.bool
            ),
        }

    def __len__(self):
        return len(self.rewards)

    def clear(self):
        self.__init__()
