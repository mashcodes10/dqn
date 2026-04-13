"""
DQN training script for Atari (Breakout / Pong).
Reproduces the core setup from Mnih et al. (2015).

Usage:
    conda activate dqn
    python train_atari.py --config configs/breakout.json --seed 0
"""

import argparse
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.wrappers import make_atari_env
from src.dqn_agent import DQNAgent


def make_eval_env(env_id, render_mode=None):
    """Create eval env: no reward clipping, no episodic life (play full games)."""
    import gymnasium as gym
    import ale_py
    from src.wrappers import (
        NoopResetWrapper, MaxAndSkipWrapper,
        GrayscaleResizeWrapper, FrameStackWrapper,
    )
    gym.register_envs(ale_py)
    env = gym.make(env_id, render_mode=render_mode)
    env = NoopResetWrapper(env, noop_max=30)
    env = MaxAndSkipWrapper(env, skip=4)
    # NO EpisodicLifeWrapper — play all 5 lives
    # NO ClipRewardWrapper — see actual game scores
    env = GrayscaleResizeWrapper(env)
    env = FrameStackWrapper(env, n_frames=4)
    return env


def evaluate(agent, env_id, n_episodes=10, render_mode=None):
    """Run evaluation episodes with greedy policy, full games, unclipped rewards."""
    env = make_eval_env(env_id, render_mode=render_mode)
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        # Press FIRE to start
        obs, _, terminated, truncated, _ = env.step(1)
        episode_reward = 0.0
        prev_lives = env.unwrapped.ale.lives()
        done = terminated or truncated
        while not done:
            with torch.no_grad():
                obs_t = torch.tensor(obs, dtype=torch.uint8, device=agent.device).unsqueeze(0)
                action = agent.policy_net(obs_t).argmax(dim=1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            # Detect life loss and press FIRE to relaunch ball
            lives = env.unwrapped.ale.lives()
            if not done and lives < prev_lives:
                obs, _, terminated, truncated, _ = env.step(1)  # FIRE
                done = terminated or truncated
            prev_lives = lives
        rewards.append(episode_reward)
    env.close()
    return np.mean(rewards), np.std(rewards)


def main():
    parser = argparse.ArgumentParser(description="Train DQN on Atari")
    parser.add_argument("--config", type=str, default="configs/breakout.json")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: cpu, cuda, or mps. Auto-detects if not set.")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    # Device selection
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Seeding
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create environment
    env_id = config["env_id"]
    env = make_atari_env(env_id)
    env.reset(seed=seed)

    n_actions = env.action_space.n
    obs_shape = env.observation_space.shape  # (4, 84, 84)
    print(f"Environment: {env_id}")
    print(f"Actions: {n_actions}, Obs shape: {obs_shape}")

    # Create agent
    agent = DQNAgent(n_actions, obs_shape, config, device=device)

    # Logging
    run_name = f"{env_id.split('/')[-1]}_seed{seed}_{int(time.time())}"
    log_dir = os.path.join("results", "runs", run_name)
    writer = SummaryWriter(log_dir)
    checkpoint_dir = os.path.join("results", "checkpoints", run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save config
    with open(os.path.join(log_dir, "config.json"), "w") as f:
        json.dump({**config, "seed": seed, "device": str(device)}, f, indent=2)

    # Training loop
    obs, _ = env.reset()
    episode_reward = 0.0
    episode_num = 0
    recent_rewards = deque(maxlen=100)
    recent_losses = deque(maxlen=100)
    start_time = time.time()

    for step in range(1, config["total_steps"] + 1):
        # Select and execute action
        action = agent.select_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store transition
        agent.store_transition(obs, action, reward, next_obs, terminated)

        # Train
        loss = agent.train_step()
        if loss is not None:
            recent_losses.append(loss)

        # Sync target network
        if step % config["target_update_frequency"] == 0:
            agent.sync_target_network()

        # Episode bookkeeping
        episode_reward += reward
        if done:
            recent_rewards.append(episode_reward)
            writer.add_scalar("train/episode_reward", episode_reward, step)
            writer.add_scalar("train/epsilon", agent.epsilon, step)
            episode_num += 1
            episode_reward = 0.0
            obs, _ = env.reset()
        else:
            obs = next_obs

        # Periodic logging
        if step % config["log_frequency"] == 0 and recent_rewards:
            elapsed = time.time() - start_time
            fps = step / elapsed
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(recent_losses) if recent_losses else 0.0
            writer.add_scalar("train/avg_reward_100", avg_reward, step)
            writer.add_scalar("train/avg_loss", avg_loss, step)
            writer.add_scalar("train/fps", fps, step)
            print(
                f"Step {step:>8d}/{config['total_steps']} | "
                f"Ep {episode_num:>5d} | "
                f"Avg100: {avg_reward:>7.2f} | "
                f"Loss: {avg_loss:.4f} | "
                f"Eps: {agent.epsilon:.3f} | "
                f"FPS: {fps:.0f}"
            )

        # Periodic evaluation
        if step % config["eval_frequency"] == 0:
            eval_mean, eval_std = evaluate(agent, env_id, n_episodes=config["eval_episodes"])
            writer.add_scalar("eval/mean_reward", eval_mean, step)
            writer.add_scalar("eval/std_reward", eval_std, step)
            print(f"  [EVAL] Step {step}: {eval_mean:.2f} +/- {eval_std:.2f}")

        # Save checkpoint
        if step % config["save_frequency"] == 0:
            path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
            agent.save(path)
            print(f"  Saved checkpoint: {path}")

    # Final save
    agent.save(os.path.join(checkpoint_dir, "final.pt"))
    env.close()
    writer.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
