"""
DQN Agent: epsilon-greedy action selection, training step, target network sync.
Follows the algorithm from Mnih et al. (2015), Algorithm 1.
"""

import torch
import torch.nn.functional as F
import numpy as np
from src.model import DQN
from src.replay_buffer import ReplayBuffer


class DQNAgent:
    def __init__(self, n_actions, obs_shape, config, device="cpu"):
        self.n_actions = n_actions
        self.device = device
        self.config = config

        # Online and target networks
        self.policy_net = DQN(n_actions, in_channels=obs_shape[0]).to(device)
        self.target_net = DQN(n_actions, in_channels=obs_shape[0]).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=config["lr"]
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(config["buffer_size"], obs_shape)

        # Epsilon schedule (linear decay)
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_decay_steps = config["epsilon_decay_steps"]
        self.steps_done = 0

    @property
    def epsilon(self):
        frac = min(1.0, self.steps_done / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs):
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        with torch.no_grad():
            obs_t = torch.tensor(obs, dtype=torch.uint8, device=self.device).unsqueeze(0)
            q_values = self.policy_net(obs_t)
            return q_values.argmax(dim=1).item()

    def store_transition(self, obs, action, reward, next_obs, done):
        self.replay_buffer.push(obs, action, reward, next_obs, done)
        self.steps_done += 1

    def train_step(self):
        """Sample a batch and perform one gradient step."""
        if len(self.replay_buffer) < self.config["learning_starts"]:
            return None

        if self.steps_done % self.config["train_frequency"] != 0:
            return None

        batch_size = self.config["batch_size"]
        gamma = self.config["gamma"]

        obs, actions, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        obs_t = torch.tensor(obs, dtype=torch.uint8, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(next_obs, dtype=torch.uint8, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q values: Q(s, a)
        q_values = self.policy_net(obs_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target Q values: r + gamma * max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_obs_t).max(dim=1)[0]
            target = rewards_t + gamma * next_q_values * (1.0 - dones_t)

        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping as in the paper
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        return loss.item()

    def sync_target_network(self):
        """Copy weights from policy net to target net."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "steps_done": self.steps_done,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.steps_done = checkpoint["steps_done"]
