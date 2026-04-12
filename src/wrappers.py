"""
Atari preprocessing wrappers following Mnih et al. (2015).
- NoopReset: random no-op actions at episode start
- FireReset: press FIRE to start games that require it
- EpisodicLife: treat life loss as episode end during training
- MaxAndSkip: repeat action for k frames, take max of last 2
- GrayScale + Resize: convert to 84x84 grayscale
- FrameStack: stack last 4 frames as input channels
"""

import gymnasium as gym
import ale_py
import numpy as np
from collections import deque

# Register ALE environments with Gymnasium
gym.register_envs(ale_py)


class NoopResetWrapper(gym.Wrapper):
    """Sample random no-ops on reset. No-op is action 0."""

    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.noop_action = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetWrapper(gym.Wrapper):
    """Press FIRE on reset for environments that require it (e.g., Breakout)."""

    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeWrapper(gym.Wrapper):
    """Make end-of-life == end-of-episode during training only."""

    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from lost life
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipWrapper(gym.Wrapper):
    """Repeat action for `skip` frames and return max of last 2 frames."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        total_reward = 0.0
        terminated = truncated = False
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer[0] = obs
        self._obs_buffer[1] = obs
        return obs, info


class GrayscaleResizeWrapper(gym.ObservationWrapper):
    """Convert to grayscale and resize to 84x84."""

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84), dtype=np.uint8
        )

    def observation(self, obs):
        # Convert to grayscale using luminance weights
        gray = np.dot(obs[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        # Resize to 84x84 using simple slicing/interpolation
        import cv2
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized


class FrameStackWrapper(gym.Wrapper):
    """Stack the last `n_frames` observations along a new first axis."""

    def __init__(self, env, n_frames=4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque([], maxlen=n_frames)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(n_frames, old_space.shape[0], old_space.shape[1]),
            dtype=np.uint8,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        return np.array(self.frames)


class ClipRewardWrapper(gym.RewardWrapper):
    """Clip rewards to {-1, 0, +1} as in the DQN paper."""

    def reward(self, reward):
        return np.sign(reward)


def make_atari_env(env_id="ALE/Breakout-v5", render_mode=None):
    """Create a fully wrapped Atari environment."""
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    env = EpisodicLifeWrapper(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetWrapper(env)
    env = GrayscaleResizeWrapper(env)
    env = ClipRewardWrapper(env)
    env = FrameStackWrapper(env, n_frames=4)
    return env
