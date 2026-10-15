#Actor, Critic Models

#πθ(a | s)  , VΦ(s)


import torch
import torch.nn as nn
from torch.distributions import Categorical


class Actor(nn.Module):
    """Policy network: state -> action distribution."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state):
        # Returns logits, not probs
        return self.network(state)

    def get_distribution(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits)


class Critic(nn.Module):
    """Value network: state -> estimated future return."""
    #How promising is the current state, in terms of expected future rewards?

    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        # Removing the final dimension: [batch, 1] -> [batch].
        return self.network(state).squeeze(-1)


class ActorCritic(nn.Module):
    """Combines the policy and value networks."""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()

        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, hidden_dim)

    def get_action_and_value(self, state, action=None):
        #During rollout, we sample an action from the policy.
        #During PPO optimization, evaluate the provided action under the current policy.

        dist = self.actor.get_distribution(state)

        # Sample an action unless one was supplied.
        if action is None:
            action = dist.sample()

        #used for calculating surrogate objective by comparing p(x)/q(x) = Rt(θ) = exp(log Pnew - log Pold)
        log_prob = dist.log_prob(action) #how likely was the selected action under the policy
        entropy = dist.entropy() #how uncertain/random or decisive is the policy's action distribution
        #equally likely action -> high entropy, decisive action -> low entropy

        value = self.critic(state)

        return action, log_prob, entropy, value
