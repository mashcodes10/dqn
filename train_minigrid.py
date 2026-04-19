import argparse
import random
import time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import minigrid
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from collections import deque
from torch.utils.tensorboard import SummaryWriter

DEFAULT_TOTAL_STEPS = 500_000
BUFFER_SIZE     = 50_000
BATCH_SIZE      = 32
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


def make_env(seed):
    env = gym.make(ENV_ID, max_steps=MAX_STEPS_EP)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env.reset(seed=seed)
    return env


class QNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 3 * 3, 512), nn.ReLU(),
            nn.Linear(512, n_actions)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = deque(maxlen=capacity)

    def push(self, obs, action, reward, next_obs, done):
        self.buf.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            torch.tensor(np.array(obs),      dtype=torch.float32),
            torch.tensor(actions,            dtype=torch.long),
            torch.tensor(rewards,            dtype=torch.float32),
            torch.tensor(np.array(next_obs), dtype=torch.float32),
            torch.tensor(dones,              dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def preprocess(obs):
    return np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0


def evaluate(env_seed_offset, policy_net, device, n_episodes=EVAL_EPISODES):
    successes = 0
    total_reward = 0.0
    for i in range(n_episodes):
        env = make_env(seed=env_seed_offset + i)
        obs, _ = env.reset()
        obs = preprocess(obs)
        ep_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                q = policy_net(
                    torch.tensor(obs[None], dtype=torch.float32).to(device)
                )
            if random.random() < 0.1:
                action = env.action_space.sample()
            else:
                action = q.argmax(1).item()
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess(obs)
            ep_reward += reward
            done = terminated or truncated
        if ep_reward > 0:
            successes += 1
        total_reward += ep_reward
        env.close()
    return total_reward / n_episodes, successes / n_episodes


def train(seed, device, total_steps=DEFAULT_TOTAL_STEPS):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"memory_baseline_seed{seed}_{int(time.time())}"
    writer = SummaryWriter(f"results/runs/{run_name}")
    print(f"[Baseline] seed={seed}  device={device}  run={run_name}")

    env = make_env(seed)
    n_actions = env.action_space.n

    policy_net = QNetwork(n_actions).to(device)
    target_net = QNetwork(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    buffer    = ReplayBuffer(BUFFER_SIZE)

    obs, _ = env.reset(seed=seed)
    obs = preprocess(obs)

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
                q = policy_net(
                    torch.tensor(obs[None], dtype=torch.float32).to(device)
                )
                action = q.argmax(1).item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = preprocess(next_obs)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, float(done))
        obs = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            writer.add_scalar("train/episode_reward", ep_reward, step)
            ep_reward = 0.0
            obs, _ = env.reset()
            obs = preprocess(obs)

        if step >= LEARNING_START and step % TRAIN_FREQ == 0 and len(buffer) >= BATCH_SIZE:
            b_obs, b_act, b_rew, b_next, b_done = buffer.sample(BATCH_SIZE)
            b_obs, b_next = b_obs.to(device), b_next.to(device)
            b_act  = b_act.to(device)
            b_rew  = b_rew.to(device)
            b_done = b_done.to(device)

            with torch.no_grad():
                target_q = b_rew + GAMMA * target_net(b_next).max(1)[0] * (1 - b_done)
            current_q = policy_net(b_obs).gather(1, b_act.unsqueeze(1)).squeeze(1)
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
                device=device
            )
            mean_loss = np.mean(losses[-500:]) if losses else 0.0
            print(
                f"  step={step:>7}  eps={eps:.3f}  "
                f"eval_reward={mean_reward:.3f}  "
                f"success_rate={success_rate:.1%}  "
                f"loss={mean_loss:.4f}"
            )
            writer.add_scalar("eval/mean_reward",  mean_reward,  step)
            writer.add_scalar("eval/success_rate", success_rate, step)
            writer.add_scalar("eval/epsilon",      eps,          step)
            writer.add_scalar("train/loss",        mean_loss,    step)

    env.close()
    writer.close()

    import os
    os.makedirs("results/checkpoints", exist_ok=True)
    torch.save(policy_net.state_dict(),
               f"results/checkpoints/memory_baseline_seed{seed}.pt")
    print(f"[Done] seed={seed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--device",      type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-steps", type=int, default=DEFAULT_TOTAL_STEPS)
    args = parser.parse_args()
    train(args.seed, args.device, args.total_steps)