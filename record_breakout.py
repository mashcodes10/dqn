# record_breakout.py
import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from src.model import DQN
from src.wrappers import (
    NoopResetWrapper, MaxAndSkipWrapper,
    GrayscaleResizeWrapper, FrameStackWrapper
)
import ale_py
gym.register_envs(ale_py)

CHECKPOINT = "results/checkpoints/Breakout-v5_seed0_1775988250/final.pt"
OUTPUT_DIR = "results/videos/breakout"
N_EPISODES = 3


def make_eval_env():
    # No EpisodicLife, no ClipReward, no FireReset for clean recording
    env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n_frames=4)
    return env


def record():
    env = make_eval_env()
    env = RecordVideo(env, OUTPUT_DIR, episode_trigger=lambda e: True,
                      name_prefix="breakout_dqn")

    model = DQN(n_actions=env.action_space.n)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu")
    model.load_state_dict(checkpoint["policy_net"])
    model.eval()

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=ep)
        done = False
        total_reward = 0.0
        while not done:
            obs_t = torch.tensor(np.array(obs), dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action = model(obs_t).argmax(1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep}: reward={total_reward:.1f}")

    env.close()
    print(f"\nVideos saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    record()