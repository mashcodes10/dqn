import argparse
import os
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


def preprocess(obs):
    return np.transpose(obs, (2, 0, 1)).astype(np.float32) / 255.0


class QNetworkRecurrent(nn.Module):
    def __init__(self, n_actions, lstm_hidden=256):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
        )
        self.lstm = nn.LSTM(64 * 3 * 3, lstm_hidden, batch_first=True)
        self.fc = nn.Linear(lstm_hidden, n_actions)

    def forward(self, x, hidden=None):
        # x: (B, T, 3, 56, 56)
        B, T = x.shape[0], x.shape[1]
        feats = self.conv(x.reshape(B * T, *x.shape[2:])).flatten(1)
        feats = feats.view(B, T, -1)
        out, hidden = self.lstm(feats, hidden)
        q = self.fc(out)  # (B, T, A)
        return q, hidden

    def init_hidden(self, batch_size, device):
        h = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        c = torch.zeros(1, batch_size, self.lstm_hidden, device=device)
        return (h, c)


class SequenceReplayBuffer:
    """Stores complete episodes; samples length-L subsequences with padding + mask."""

    def __init__(self, capacity_episodes, seq_len):
        self.episodes = deque(maxlen=capacity_episodes)
        self.seq_len = seq_len
        self.current = []

    def push(self, obs, action, reward, next_obs, done):
        self.current.append((obs, action, reward, next_obs, float(done)))
        if done:
            if len(self.current) > 0:
                self.episodes.append(self.current)
            self.current = []

    def __len__(self):
        return len(self.episodes)

    def sample(self, batch_size):
        L = self.seq_len
        obs_shape = self.episodes[0][0][0].shape  # (3, 56, 56)

        b_obs      = np.zeros((batch_size, L, *obs_shape), dtype=np.float32)
        b_next_obs = np.zeros((batch_size, L, *obs_shape), dtype=np.float32)
        b_act      = np.zeros((batch_size, L), dtype=np.int64)
        b_rew      = np.zeros((batch_size, L), dtype=np.float32)
        b_done     = np.zeros((batch_size, L), dtype=np.float32)
        b_mask     = np.zeros((batch_size, L), dtype=np.float32)

        for i in range(batch_size):
            ep = random.choice(self.episodes)
            ep_len = len(ep)
            if ep_len <= L:
                start = 0
                take = ep_len
            else:
                start = random.randint(0, ep_len - L)
                take = L
            for t in range(take):
                o, a, r, no, d = ep[start + t]
                b_obs[i, t]      = o
                b_next_obs[i, t] = no
                b_act[i, t]      = a
                b_rew[i, t]      = r
                b_done[i, t]     = d
                b_mask[i, t]     = 1.0
        return (
            torch.from_numpy(b_obs),
            torch.from_numpy(b_act),
            torch.from_numpy(b_rew),
            torch.from_numpy(b_next_obs),
            torch.from_numpy(b_done),
            torch.from_numpy(b_mask),
        )


def evaluate(env_seed_offset, policy_net, device, n_episodes=EVAL_EPISODES, eps_eval=0.0):
    successes = 0
    total_reward = 0.0
    for i in range(n_episodes):
        env = make_env(seed=env_seed_offset + i)
        obs, _ = env.reset()
        obs = preprocess(obs)
        hidden = policy_net.init_hidden(1, device)
        ep_reward = 0.0
        done = False
        while not done:
            with torch.no_grad():
                x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
                q, hidden = policy_net(x, hidden)
                if random.random() < eps_eval:
                    action = env.action_space.sample()
                else:
                    action = q[0, 0].argmax().item()
            obs, reward, terminated, truncated, _ = env.step(action)
            obs = preprocess(obs)
            ep_reward += reward
            done = terminated or truncated
        if ep_reward > 0:
            successes += 1
        total_reward += ep_reward
        env.close()
    return total_reward / n_episodes, successes / n_episodes


def train(args):
    seed       = args.seed
    device     = args.device
    total_steps = args.total_steps
    seq_len    = args.seq_len
    lstm_hid   = args.lstm_hidden
    batch_size = args.batch_size
    buffer_eps = args.buffer_episodes

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    run_name = f"memory_drqn_seed{seed}_L{seq_len}_{int(time.time())}"
    writer = SummaryWriter(f"results/runs/{run_name}")
    print(f"[DRQN] seed={seed}  device={device}  total_steps={total_steps}  "
          f"seq_len={seq_len}  lstm_hidden={lstm_hid}  run={run_name}")

    env = make_env(seed)
    n_actions = env.action_space.n

    policy_net = QNetworkRecurrent(n_actions, lstm_hidden=lstm_hid).to(device)
    target_net = QNetworkRecurrent(n_actions, lstm_hidden=lstm_hid).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(policy_net.parameters(), lr=LR)
    buffer = SequenceReplayBuffer(buffer_eps, seq_len)

    obs, _ = env.reset(seed=seed)
    obs = preprocess(obs)
    hidden = policy_net.init_hidden(1, device)

    eps       = EPS_START
    ep_reward = 0.0
    ep_count  = 0
    losses    = []

    for step in range(1, total_steps + 1):

        eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * step / EPS_DECAY_STEPS)

        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
            q, hidden = policy_net(x, hidden)
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = q[0, 0].argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_obs = preprocess(next_obs)
        done = terminated or truncated

        buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        ep_reward += reward

        if done:
            ep_count += 1
            writer.add_scalar("train/episode_reward", ep_reward, step)
            ep_reward = 0.0
            obs, _ = env.reset()
            obs = preprocess(obs)
            hidden = policy_net.init_hidden(1, device)

        if step >= LEARNING_START and step % TRAIN_FREQ == 0 and len(buffer) >= batch_size:
            b_obs, b_act, b_rew, b_next, b_done, b_mask = buffer.sample(batch_size)
            b_obs  = b_obs.to(device)
            b_next = b_next.to(device)
            b_act  = b_act.to(device)
            b_rew  = b_rew.to(device)
            b_done = b_done.to(device)
            b_mask = b_mask.to(device)

            q_seq, _ = policy_net(b_obs, None)
            current_q = q_seq.gather(-1, b_act.unsqueeze(-1)).squeeze(-1)

            with torch.no_grad():
                q_next, _ = target_net(b_next, None)
                target_q = b_rew + GAMMA * q_next.max(-1)[0] * (1.0 - b_done)

            td = nn.functional.huber_loss(current_q, target_q, reduction="none")
            loss = (td * b_mask).sum() / b_mask.sum().clamp(min=1.0)

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
    ckpt_path = f"results/checkpoints/memory_drqn_seed{seed}_L{seq_len}.pt"
    torch.save(policy_net.state_dict(), ckpt_path)
    print(f"[Done] seed={seed}  checkpoint={ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",            type=int, default=0)
    parser.add_argument("--device",          type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-steps",     type=int, default=2_000_000)
    parser.add_argument("--seq-len",         type=int, default=10)
    parser.add_argument("--lstm-hidden",     type=int, default=256)
    parser.add_argument("--batch-size",      type=int, default=16)
    parser.add_argument("--buffer-episodes", type=int, default=2_000)
    args = parser.parse_args()
    train(args)
