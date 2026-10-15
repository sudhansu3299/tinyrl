
import torch
import torch.nn as nn


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, state, action):
        # Concatenate state and action along the feature dimension
        state_action = torch.cat([state, action], dim=-1)
        q_value = self.network(state_action)

        return q_value
