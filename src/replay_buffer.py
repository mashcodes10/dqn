"""
Experience Replay Buffer for DQN (Mnih et al., 2015).
Stores transitions (s, a, r, s', done) and samples uniform random minibatches.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.idx = 0
        self.size = 0

        # Pre-allocate arrays for memory efficiency
        self.observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, obs, action, reward, next_obs, done):
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = done

        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size
