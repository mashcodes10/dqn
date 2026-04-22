"""
Frame-stacking DQN on MiniGrid-MemoryS7-v0 (Phase-2 ablation).

Same feed-forward CNN as the baseline DQN, but feeds the last K frames
stacked along the channel axis so the agent has the same context window
(K=10) as the DRQN during training. If memory *per se* is what matters,
this should still fail; if extra context alone is enough, it should
succeed. This isolates the LSTM recurrence as the design change.

Runs on CUDA, Apple MPS, or CPU.
"""

import argparse
import os
import random
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid  # noqa: F401 — registers MiniGrid envs
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from torch.utils.tensorboard import SummaryWriter

GAMMA           = 0.99
LR              = 1e-4
EPS_START       = 1.0
EPS_END         = 0.1
EPS_DECAY_STEPS = 150_000
TARGET_UPDATE   = 1_000
LEARNING_START  = 2_000
TRAIN_FREQ      = 4
EVAL_FREQ       = 10_000
EVAL_EPISODES   = 100
MAX_STEPS_EP    = 300
ENV_ID          = "MiniGrid-MemoryS7-v0"


def pick_device(arg):
    if arg:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_env(seed):
    env = gym.make(ENV_ID, max_steps=MAX_STEPS_EP)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)
    return env


def frame_to_chw(obs):
    # (56, 56, 3) uint8 -> (3, 56, 56) uint8
    return np.transpose(obs, (2, 0, 1))


class FrameStack:
    """Holds the last K frames; returns them stacked along channels."""

    def __init__(self, k):
        self.k = k
        self.buf = deque(maxlen=k)

    def reset(self, first_frame):
        self.buf.clear()
        for _ in range(self.k):
            self.buf.append(first_frame)
        return self._stack()

    def step(self, frame):
        self.buf.append(frame)
        return self._stack()

    def _stack(self):
        # list of K frames (3, H, W) uint8 -> (3*K, H, W) uint8
        return np.concatenate(list(self.buf), axis=0)


class QNetwork(nn.Module):
    def __init__(self, n_actions, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 512), nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def forward(self, x):
        # x is uint8 (B, 3K, 56, 56); scale to [0,1] inside the net
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class ReplayBuffer:
    """Stores uint8 stacked observations to keep memory reasonable."""

    def __init__(self, capacity, obs_shape):
        self.capacity = capacity
        self.idx = 0
        self.size = 0
        self.obs      = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.uint8)
        self.act      = np.zeros(capacity, dtype=np.int64)
        self.rew      = np.zeros(capacity, dtype=np.float32)
        self.done     = np.zeros(capacity, dtype=np.float32)

    def push(self, obs, action, reward, next_obs, done):
        self.obs[self.idx]      = obs
        self.next_obs[self.idx] = next_obs
        self.act[self.idx]      = action
        self.rew[self.idx]      = reward
        self.done[self.idx]     = float(done)
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        ii = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.from_numpy(self.obs[ii]),
            torch.from_numpy(self.act[ii]),
            torch.from_numpy(self.rew[ii]),
            torch.from_numpy(self.next_obs[ii]),
            torch.from_numpy(self.done[ii]),
        )

    def __len__(self):
        return self.size


def evaluate(env_seed_offset, policy_net, device, stack_k, n_episodes=EVAL_EPISODES, eps_eval=0.0):
    successes = 0
    total_reward = 0.0
    for i in range(n_episodes):
        env = make_env(seed=env_seed_offset + i)
        raw, _ = env.reset()
        stacker = FrameStack(stack_k)
        obs = stacker.reset(frame_to_chw(raw))
        ep_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                x = torch.from_numpy(obs).unsqueeze(0).to(device)
                q = policy_net(x)
                if random.random() < eps_eval:
                    action = env.action_space.sample()
                else:
                    action = q.argmax(1).item()
            raw, reward, terminated, truncated, _ = env.step(action)
            obs = stacker.step(frame_to_chw(raw))
            ep_reward += reward
            done = terminated or truncated
        if ep_reward > 0:
            successes += 1
        total_reward += ep_reward
        env.close()
    return total_reward / n_episodes, successes / n_episodes


def train(args):
    seed       = args.seed
    device     = pick_device(args.device)
    total_steps = args.total_steps
    stack_k    = args.stack_k
    batch_size = args.batch_size
    buffer_sz  = args.buffer_size

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"memory_framestack_seed{seed}_K{stack_k}_{int(time.time())}"
    writer = SummaryWriter(f"results/runs/{run_name}")
    print(f"[FrameStack] seed={seed}  device={device}  total_steps={total_steps}  "
          f"K={stack_k}  run={run_name}")

    env = make_env(seed)
    n_actions = env.action_space.n
    in_channels = 3 * stack_k
    obs_shape = (in_channels, 56, 56)

    policy_net = QNetwork(n_actions, in_channels).to(device)
    target_net = QNetwork(n_actions, in_channels).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(buffer_sz, obs_shape)

    raw, _ = env.reset(seed=seed)
    stacker = FrameStack(stack_k)
    obs = stacker.reset(frame_to_chw(raw))

    eps       = EPS_START
    ep_reward = 0.0
    ep_count  = 0
    losses    = []

    for step in range(1, total_steps + 1):
        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY_STEPS)

        if random.random() < eps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                x = torch.from_numpy(obs).unsqueeze(0).to(device)
                action = policy_net(x).argmax(1).item()

        raw, reward, terminated, truncated, _ = env.step(action)
        next_obs = stacker.step(frame_to_chw(raw))
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            writer.add_scalar("train/episode_reward", ep_reward, step)
            ep_reward = 0.0
            raw, _ = env.reset()
            obs = stacker.reset(frame_to_chw(raw))

        if step >= LEARNING_START and step % TRAIN_FREQ == 0 and len(buffer) >= batch_size:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(batch_size)
            b_obs  = b_obs.to(device)
            b_next = b_next.to(device)
            b_act  = b_act.to(device)
            b_rew  = b_rew.to(device)
            b_done = b_done.to(device)

            current_q = policy_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                target_q = b_rew + GAMMA * target_net(b_next).max(1)[0] * (1.0 - b_done)

            loss = nn.functional.huber_loss(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy_net.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.item())

        if step % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % EVAL_FREQ == 0:
            mean_reward, success_rate = evaluate(
                env_seed_offset=10000 + seed * 1000,
                policy_net=policy_net,
                device=device,
                stack_k=stack_k,
            )
            mean_loss = float(np.mean(losses[-500:])) if losses else 0.0
            print(
                f"  step={step:>7}  eps={eps:.3f}  "
                f"eval_reward={mean_reward:.3f}  "
                f"success_rate={success_rate:.1%}  "
                f"loss={mean_loss:.4f}  "
                f"episodes={ep_count}"
            )
            writer.add_scalar("eval/mean_reward",  mean_reward,  step)
            writer.add_scalar("eval/success_rate", success_rate, step)
            writer.add_scalar("eval/epsilon",      eps,          step)
            writer.add_scalar("train/loss",        mean_loss,    step)

    env.close()
    writer.close()

    os.makedirs("results/checkpoints", exist_ok=True)
    ckpt_path = f"results/checkpoints/memory_framestack_seed{seed}_K{stack_k}.pt"
    torch.save(policy_net.state_dict(), ckpt_path)
    print(f"[Done] seed={seed}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--device",      type=str, default=None,
                        help="cuda / mps / cpu. Auto-detected if omitted.")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--stack-k",     type=int, default=10,
                        help="Number of past frames to stack (match DRQN seq_len).")
    parser.add_argument("--batch-size",  type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=20_000,
                        help="Smaller than baseline because each obs is K-channel stacked.")
    args = parser.parse_args()
    train(args)
