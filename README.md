# DQN: Reproducing and Extending Classical Reinforcement Learning

**Track 1: DQN** — Reproducing Mnih et al. (2015) and adapting to a memory-dependent environment.

## Project Overview

This project has two phases:

| Phase | Environment | Goal | Status |
|-------|------------|------|--------|
| **Phase 1: Reproduction** | ALE Breakout (Atari) | Train DQN from pixels, show it learns | In Progress |
| **Phase 2: Adaptation** | MiniGrid MemoryEnv | Add recurrent memory (DRQN) to handle partial observability | Not Started |

## Paper Reference

**Human-level control through deep reinforcement learning** — Mnih et al., Nature, 2015

Key ideas reproduced:
- CNN Q-network learning directly from raw pixels (84x84 grayscale)
- Experience replay buffer for stable training
- Separate target network (hard-copied every 10k steps)
- Epsilon-greedy exploration with linear decay
- Reward clipping to {-1, 0, +1}
- Frame stacking (4 frames), frame skipping (4), max over last 2 frames

## Repository Structure

```
├── src/
│   ├── model.py            # CNN Q-network (3 conv + 2 FC, matches paper Table 1)
│   ├── dqn_agent.py        # DQN agent (epsilon-greedy, training, target sync)
│   ├── replay_buffer.py    # Uniform experience replay
│   └── wrappers.py         # Atari preprocessing (NoOp, MaxSkip, EpisodicLife, etc.)
├── configs/
│   ├── breakout.json       # Current active config (10M steps)
│   ├── breakout_5m.json    # 5M step config
│   └── breakout_10m.json   # 10M step config
├── train_atari.py          # Main training script with TensorBoard logging
├── plot_results.py         # Script to generate result plots
├── DQN_Breakout_Colab.ipynb # Google Colab notebook (Drive checkpoints + auto-resume)
├── results/
│   ├── runs/               # TensorBoard logs
│   └── checkpoints/        # Model checkpoints (.pt files)
└── Final_Project.pdf       # Project assignment specification
```

## Setup Instructions

### Local (Mac / Linux)

```bash
# Create environment
conda create -n dqn python=3.11 -y
conda activate dqn

# Install dependencies
pip install torch torchvision gymnasium "gymnasium[atari]" ale-py autorom \
    minigrid tensorboard matplotlib numpy opencv-python-headless

# Download Atari ROMs
AutoROM --accept-license

# Verify setup
python -c "
import gymnasium as gym; import ale_py; gym.register_envs(ale_py)
env = gym.make('ALE/Breakout-v5'); obs, _ = env.reset()
print('Breakout OK:', obs.shape)
env.close()
"
```

### Google Colab

Upload `DQN_Breakout_Colab.ipynb` to Colab. It handles all installation, saves checkpoints to Google Drive, and auto-resumes after disconnects.

## Training

### Run locally
```bash
conda activate dqn
python train_atari.py --config configs/breakout_10m.json --seed 0
```

### Run all 3 seeds (chained)
```bash
python train_atari.py --config configs/breakout_10m.json --seed 0 && \
python train_atari.py --config configs/breakout_10m.json --seed 1 && \
python train_atari.py --config configs/breakout_10m.json --seed 2
```

### Monitor training
```bash
tensorboard --logdir results/runs
```

## Hyperparameters

| Parameter | Our Value | Paper Value | Notes |
|-----------|----------|-------------|-------|
| Total steps | 10M (=40M frames) | 50M frames | Scaled down for feasibility |
| Replay buffer | 200,000 | 1,000,000 | Reduced for memory constraints |
| Batch size | 32 | 32 | Same |
| Discount (γ) | 0.99 | 0.99 | Same |
| Learning rate | 1e-4 (Adam) | 2.5e-4 (RMSProp) | Different optimizer |
| Epsilon start | 1.0 | 1.0 | Same |
| Epsilon end | 0.1 | 0.1 | Same |
| Epsilon decay | 1M steps | 1M frames (~250k steps) | Ours decays 4x slower |
| Target update freq | 10,000 steps | 10,000 steps | Same |
| Frame skip | 4 | 4 | Same |
| Frame stack | 4 | 4 | Same |
| Input size | 84x84 grayscale | 84x84 grayscale | Same |

## Results So Far

### Phase 1: Breakout Reproduction

**Seed 0 (2M steps, Mac MPS):**
- Eval reward (full game, unclipped): ~33 mean
- Clear learning curve from ~0 to ~33
- Note: early runs had eval bugs (clipped rewards + single-life episodes) that under-reported scores

**Seed 0 (10M steps, CUDA):**
- Eval reward: ~40-50, peaks to ~110
- Strong upward trend throughout training

**Seeds 1 & 2:** Running on CUDA (10M steps each)

**Comparison to paper:**

| Agent | Breakout Score |
|-------|---------------|
| Random | 1.7 |
| Human | 31.8 |
| **Our DQN (10M steps)** | **~40-50** |
| Paper DQN (50M frames) | 401.2 |

Our agent beats human-level at 10M steps. The gap to the paper's 401 is explained by: smaller replay buffer (200k vs 1M), slower epsilon decay, different optimizer (Adam vs RMSProp), and fewer training frames.

### Phase 2: Adaptation (Planned)

- [ ] Run baseline DQN on MiniGrid MemoryEnv → expect ~50% success (random guessing)
- [ ] Implement DRQN (DQN + LSTM) with sequence replay buffer
- [ ] Train DRQN on MemoryEnv → expect success rate well above 50%
- [ ] Run ablation (e.g., sequence length, LSTM vs no-LSTM)

## Known Issues & Divergences from Paper

1. **Eval reporting bug (fixed):** Early runs used reward clipping and EpisodicLifeWrapper during evaluation, which reported artificially low scores (~5-7 instead of actual ~33). Fixed in commit `7b3d853`.

2. **Replay buffer size:** Paper uses 1M transitions. We use 200k due to RAM constraints. This limits training diversity.

3. **Epsilon decay mismatch:** Our epsilon decays over 1M agent steps (= 4M environment frames due to frame skip). The paper decays over 1M environment frames (= 250k agent steps). This means our agent explores randomly for 4x longer.

4. **Optimizer:** Paper uses RMSProp with specific settings (lr=0.00025, α=0.95, ε=0.01). We use Adam (lr=1e-4). Both are valid but produce different learning dynamics.

## Environment Versions

- Python: 3.11
- PyTorch: 2.11.0
- Gymnasium: 1.2.3
- ALE-py: 0.11.2
- MiniGrid: 3.0.0

## Compute Budget

| Run | Device | Steps | Approx Time |
|-----|--------|-------|-------------|
| Seed 0 (2M) | Apple M4 MPS | 2M | ~3 hours |
| Seed 0 (10M) | NVIDIA T4 CUDA | 10M | ~5 hours |
| Seed 1 (10M) | NVIDIA T4 CUDA | 10M | ~5 hours |
| Seed 2 (10M) | NVIDIA T4 CUDA | 10M | ~5 hours |
