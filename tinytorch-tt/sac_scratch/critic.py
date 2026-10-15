
import torch
import torch.nn as nn

from cnn_feature_extractor import CNNFeatureExtractor


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, use_cnn=False):
        super().__init__()

        self.use_cnn = use_cnn
        
        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(input_channels=3, feature_dim=256)
            self.network = nn.Sequential(
                nn.Linear(256 + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(state_dim + action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1),
            )

    def forward(self, state, action):
        if self.use_cnn:
            state_features = self.feature_extractor(state)
            state_action = torch.cat([state_features, action], dim=-1)
        else:
            # Concatenate state and action along the feature dimension
            state_action = torch.cat([state, action], dim=-1)
        q_value = self.network(state_action)

        return q_value
