
import torch
import torch.nn as nn
from torch.distributions import Normal

from cnn_feature_extractor import CNNFeatureExtractor


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=2.0, use_cnn=False):
        super().__init__()

        self.use_cnn = use_cnn
        
        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(input_channels=3, feature_dim=256)
            self.backbone = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
            )
        else:
            self.backbone = nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )

        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        self.action_scale = action_scale

    def forward(self, state):
        if self.use_cnn:
            features = self.feature_extractor(state)
            features = self.backbone(features)
        else:
            features = self.backbone(state)

        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)

        # clamping keeps the std numerically stable from extreme values
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()

        return mean, std #instead of action , we are getting a dist

    def sample(self, state):
        mean, std = self.forward(state)

        distribution = Normal(mean, std) #Gaussian dist for each dim

        # Reparameterized sampling: gradients flow through the sample
        # gives us a random value from gaussian, but rsample uses reparametrization => u = mean + std * epsilon ~ N(0,1)
        # rsample() is preferred over sample() as it is differentiable
        u = distribution.rsample()

        squashed_action = torch.tanh(u) #need to squash betn [-1,1] for bounded actions
        #when we transform a random variable, its probability density changes too. Large parts of the distribution gets squashed to borders
        action = squashed_action * self.action_scale

        # Log probability density
        # SAC optimizes a trade-off between reward and entropy in actor's objective, thus needs log_prob
        log_prob = distribution.log_prob(u)

        # tanh squashing correction
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6) # this is a small stabilizer so that we don't take log of 0 as d(tanh) ~0 near y = -1/1

        #action scale correction
        log_prob -= torch.log(
            torch.tensor(self.action_scale, device=state.device)
        )

        # sum log probabilities for each action dimension as 2 conditionally independent actions
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
