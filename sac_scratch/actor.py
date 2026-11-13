
import torch
import torch.nn as nn
from torch.distributions import Normal

from cnn_feature_extractor import CNNFeatureExtractor


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_scale=2.0, use_cnn=False, action_low=None, action_high=None):
        super().__init__()

        self.use_cnn = use_cnn
        
        if use_cnn:
            self.feature_extractor = CNNFeatureExtractor(input_channels=4, feature_dim=256)
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

        # Support per-dimension action bounds (e.g., CarRacing: steering [-1,1], gas [0,1], brake [0,1])
        if action_low is not None and action_high is not None:
            self.action_low = torch.tensor(action_low, dtype=torch.float32)
            self.action_high = torch.tensor(action_high, dtype=torch.float32)
            self.action_scale = None  # ignored when per-dim bounds are provided
        else:
            self.action_low = None
            self.action_high = None
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

    def _squash_to_bounds(self, squashed):
        """Map tanh output [-1,1] to the actual action bounds."""
        if self.action_low is not None and self.action_high is not None:
            # Map [-1, 1] -> [action_low, action_high]
            # scale = (high - low) / 2, mid = (high + low) / 2
            low = self.action_low.to(squashed.device)
            high = self.action_high.to(squashed.device)
            scale = (high - low) / 2.0
            mid = (high + low) / 2.0
            return squashed * scale + mid
        else:
            return squashed * self.action_scale

    def sample(self, state):
        mean, std = self.forward(state)

        distribution = Normal(mean, std) #Gaussian dist for each dim

        # Reparameterized sampling: gradients flow through the sample
        # gives us a random value from gaussian, but rsample uses reparametrization => u = mean + std * epsilon ~ N(0,1)
        # rsample() is preferred over sample() as it is differentiable
        u = distribution.rsample()

        squashed_action = torch.tanh(u) #need to squash betn [-1,1] for bounded actions
        #when we transform a random variable, its probability density changes too. Large parts of the distribution gets squashed to borders
        action = self._squash_to_bounds(squashed_action)

        # Log probability density
        # SAC optimizes a trade-off between reward and entropy in actor's objective, thus needs log_prob
        log_prob = distribution.log_prob(u)

        # tanh squashing correction
        log_prob -= torch.log(1 - squashed_action.pow(2) + 1e-6) # this is a small stabilizer so that we don't take log of 0 as d(tanh) ~0 near y = -1/1

        # action scale correction (Jacobian of the bounds transformation)
        if self.action_low is not None and self.action_high is not None:
            scale = (self.action_high - self.action_low) / 2.0
            log_prob -= torch.log(scale.to(state.device))
        else:
            log_prob -= torch.log(
                torch.tensor(self.action_scale, device=state.device)
            )

        # sum log probabilities for each action dimension as 2 conditionally independent actions
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
