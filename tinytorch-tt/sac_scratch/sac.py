
import copy

import torch
import torch.nn as nn
import torch.optim as optim

import math

from actor import Actor
from critic import Critic


class SAC:
    def __init__(
            self,
            state_dim,
            action_dim,
            action_scale=2.0,
            gamma=0.99,
            tau=0.005, # soft update coeff (how much target critic params are updated towards online critic)
            alpha=0.2, # entropy coeff/temp.
            lr=3e-4,
            device=None,
            auto_entropy_tuning=True,
            use_cnn=False,
    ):
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.gamma = gamma
        self.tau = tau

        self.auto_entropy_tuning = auto_entropy_tuning
        self.log_alpha = torch.tensor(
            math.log(alpha), # make it log so that later exp would make it +ve, even tho after learning log alpha is -ve
            dtype=torch.float32,
            device=self.device,
            requires_grad=auto_entropy_tuning,
        )

        self.alpha = alpha
        self.target_entropy = -float(action_dim) #policy randomness as per action space

        self.alpha_optimizer = (
            optim.Adam([self.log_alpha], lr=lr)
            if auto_entropy_tuning
            else None
        )

        self.actor = Actor(
            state_dim, action_dim, action_scale, use_cnn=use_cnn
        ).to(self.device)

        # two Q networks Q1 and Q2 initialized
        # helps in controlling the overestimation bias of Q values
        self.critic1 = Critic(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        self.critic2 = Critic(state_dim, action_dim, use_cnn=use_cnn).to(self.device)
        '''Why not more than 2? because of law of diminishing returns, pessimistic trap(actor doesn't explore enough), massive computational overhead'''

        # Target critics start with identical parameters.
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)

        # target critics are used only to calculate targets as they give out target Q values
        for param in self.target_critic1.parameters():
            param.requires_grad = False

        for param in self.target_critic2.parameters():
            param.requires_grad = False

        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=lr
        )
        self.critic1_optimizer = optim.Adam(
            self.critic1.parameters(), lr=lr
        )
        self.critic2_optimizer = optim.Adam(
            self.critic2.parameters(), lr=lr
        )

    @torch.no_grad()
    def select_action(self, state, evaluate=False):
        state = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0) # as actor expects a batch of states

        if evaluate:
            # Deterministic action for evaluation.
            mean, _ = self.actor(state)
            action = torch.tanh(mean) * self.actor.action_scale # to squash action from the sample to -1,1
        else:
            action, _ = self.actor.sample(state)

        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = (
            replay_buffer.sample(batch_size)
        ) # batch of transitions

        states = torch.as_tensor(
            states, dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            actions, dtype=torch.float32, device=self.device
        )
        rewards = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        next_states = torch.as_tensor(
            next_states, dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            dones, dtype=torch.float32, device=self.device
        )

        # 1. Calculate the Bellman target.
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(
                next_states
            ) #target don't need gradient flow

            target_q1 = self.target_critic1(
                next_states, next_actions
            )
            target_q2 = self.target_critic2(
                next_states, next_actions
            )

            min_target_q = torch.minimum(target_q1, target_q2)

            target_q = rewards + self.gamma * (1 - dones) * (
                    min_target_q - self.alpha * next_log_probs
            ) # y = r + gamma * (1 - done) * (Q(s', a') - alpha * log pi(a' | s'))
            # what you earn + what you expect to earn from future(discounted)

        # 2. Update Critic 1.
        current_q1 = self.critic1(states, actions)
        critic1_loss = nn.functional.mse_loss(
            current_q1, target_q
        ) # Loss = (Q(s,a), target_q)^2

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # 3. Update Critic 2.
        current_q2 = self.critic2(states, actions)
        critic2_loss = nn.functional.mse_loss(
            current_q2, target_q
        )

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # 4. Update Actor.
        sampled_actions, log_probs = self.actor.sample(states) # new current action from current policy

        # Freeze critic parameters during actor optimization as to prevent unnecesaary gradient calculation for critic
        # Gradients still flow through Q with respect to the actions dq/da and actor learns from feedback
        for p in self.critic1.parameters():
            p.requires_grad = False
        for p in self.critic2.parameters():
            p.requires_grad = False

        q1_pi = self.critic1(states, sampled_actions) # Q1(s,a)
        q2_pi = self.critic2(states, sampled_actions) # Q2(s,a)
        min_q_pi = torch.minimum(q1_pi, q2_pi) #Q(s,a)

        actor_loss = (
                self.alpha * log_probs - min_q_pi
        ).mean() # actor objective loss = E[alpha * log pi(a|s) - Q(s,a)], -ve Q(s,a) because it would minimize the maximized Q value i.e.favorable action

        #update the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step() #critic's weights remain unchanged during this optimization step

        # restore critic gradients as next step we want the critics to learn
        for p in self.critic1.parameters():
            p.requires_grad = True
        for p in self.critic2.parameters():
            p.requires_grad = True

        # 5. Update entropy temperature.
        alpha_loss_value = 0.0

        if self.auto_entropy_tuning:
            alpha_loss = -(
                    self.log_alpha
                    * (log_probs.detach() + self.target_entropy)
            ).mean() # entropy loss = -alpha * (log pi(a|s) + target_entropy)

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp().item()
            alpha_loss_value = alpha_loss.item()


    # 6. Soft-update target critics.
        with torch.no_grad():
            for target, online in zip(
                    self.target_critic1.parameters(),
                    self.critic1.parameters(),
            ):
                target.lerp_(online, self.tau) # target = target + tau * (online - target)

            for target, online in zip(
                    self.target_critic2.parameters(),
                    self.critic2.parameters(),
            ):
                target.lerp_(online, self.tau)

        return {
            "actor_loss": actor_loss.item(),
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss_value,
        }

    def save_checkpoint(self, path, global_step=0, extra=None):
        from pathlib import Path

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "global_step": global_step,
            "actor": self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
            "target_critic1": self.target_critic1.state_dict(),
            "target_critic2": self.target_critic2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic1_optimizer": self.critic1_optimizer.state_dict(),
            "critic2_optimizer": self.critic2_optimizer.state_dict(),
            "alpha": self.alpha,
            "extra": extra or {},
        }

        if getattr(self, "auto_entropy_tuning", False):
            checkpoint["log_alpha"] = self.log_alpha.detach().cpu()
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()

        torch.save(checkpoint, path)


    def load_checkpoint(self, path, map_location=None):
        checkpoint = torch.load(
            path,
            map_location=map_location or self.device,
            weights_only=False,
        )

        self.actor.load_state_dict(checkpoint["actor"])
        self.critic1.load_state_dict(checkpoint["critic1"])
        self.critic2.load_state_dict(checkpoint["critic2"])
        self.target_critic1.load_state_dict(checkpoint["target_critic1"])
        self.target_critic2.load_state_dict(checkpoint["target_critic2"])

        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        self.critic2_optimizer.load_state_dict(checkpoint["critic2_optimizer"])

        self.alpha = float(checkpoint["alpha"])

        if "log_alpha" in checkpoint and getattr(
                self, "auto_entropy_tuning", False
        ):
            with torch.no_grad():
                self.log_alpha.copy_(
                    checkpoint["log_alpha"].to(self.device)
                )
            if "alpha_optimizer" in checkpoint:
                self.alpha_optimizer.load_state_dict(
                    checkpoint["alpha_optimizer"]
                )

        return checkpoint
