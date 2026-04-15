# DQN: Reproducing and Extending Classical Reinforcement Learning

**Track 1: DQN** — Reproducing Mnih et al. (2015) and adapting to a memory-dependent environment.

## Project Overview

This project has two phases:

| Phase | Environment | Goal | Status |
|-------|------------|------|--------|
| **Phase 1: Reproduction** | ALE Breakout (Atari) | Train DQN from pixels, show it learns | Complete |
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
│   ├── breakout.json       # Base config
│   ├── breakout_5m.json    # 5M step config
│   └── breakout_10m.json   # 10M step config (used for final results)
├── train_atari.py          # Main training script with TensorBoard logging
├── plot_results.py         # Script to generate result plots
├── DQN_Breakout_Colab.ipynb # Google Colab notebook (Drive checkpoints + auto-resume)
├── results/
│   ├── runs/               # TensorBoard logs
│   ├── checkpoints/        # Model checkpoints (.pt files)
│   └── training_3seeds.png # Final results plot
└── Final_Project.pdf       # Project assignment specification
```

## Setup Instructions

### Local (Windows, NVIDIA GPU)

```bash
conda create -n dqn python=3.11 -y
conda activate dqn
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "gymnasium[atari]" ale-py opencv-python tensorboard matplotlib
```

### Local (Mac / Linux)

```bash
conda create -n dqn python=3.11 -y
conda activate dqn
pip install torch torchvision gymnasium "gymnasium[atari]" ale-py autorom \
    minigrid tensorboard matplotlib numpy opencv-python-headless
AutoROM --accept-license
```

### Google Colab

Upload `DQN_Breakout_Colab.ipynb` to Colab. It handles all installation, saves checkpoints to Google Drive, and auto-resumes after disconnects.

## Training

```bash
conda activate dqn

# Single run
python train_atari.py --config configs/breakout_10m.json --seed 0 --device cuda

# Run all 3 seeds (chained)
python train_atari.py --config configs/breakout_10m.json --seed 0 --device cuda && \
python train_atari.py --config configs/breakout_10m.json --seed 1 --device cuda && \
python train_atari.py --config configs/breakout_10m.json --seed 2 --device cuda

# Monitor
tensorboard --logdir results/runs

# Plot
python plot_results.py
```

## Hyperparameters

| Parameter | Our Value | Paper Value | Notes |
|-----------|----------|-------------|-------|
| Total steps | 10M (=40M frames) | 50M frames | Scaled down for feasibility |
| Replay buffer | 200,000 | 1,000,000 | Reduced for memory constraints (~32 GB RAM) |
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

## Results

### Phase 1: Breakout Reproduction (Complete)

All 3 seeds trained for 10M steps on NVIDIA RTX PRO 500 Blackwell GPU (~165 FPS).

| Seed | Final Eval Reward | Peak Eval Reward |
|------|------------------|-----------------|
| 0 | 55.9 | 64.7 |
| 1 | 54.4 | 63.7 |
| 2 | 45.8 | 60.5 |
| **Mean ± Std** | **52.0 ± 5.3** | — |

![Training Results](results/training_3seeds.png)

- **Top**: Mean ± std eval reward across 3 seeds (unclipped, full 5-life games)
- **Bottom left**: Individual seed curves
- **Bottom right**: 100-episode rolling average training reward (clipped)

**Comparison to paper:**

| Agent | Breakout Score |
|-------|---------------|
| Random | 1.7 |
| Human | 31.8 |
| **Our DQN (10M steps, mean)** | **52.0 ± 5.3** |
| Paper DQN (50M frames) | 401.2 |

Our agent surpasses human-level performance. The gap to the paper's 401 is explained by: smaller replay buffer (200k vs 1M), fewer training frames (10M vs 50M agent steps), and slower epsilon decay.

### Phase 2: Adaptation (Planned)

- [ ] Run baseline DQN on MiniGrid MemoryEnv → expect ~50% success (random guessing)
- [ ] Implement DRQN (DQN + LSTM) with sequence replay buffer
- [ ] Train DRQN on MemoryEnv → expect success rate well above 50%
- [ ] Run ablation (e.g., sequence length, LSTM vs no-LSTM)

## Known Issues & Divergences from Paper

1. **Eval reporting bug (fixed):** Early runs used reward clipping and EpisodicLifeWrapper during evaluation, which reported artificially low scores (~5-7 instead of actual ~50+). Fixed in commit `7b3d853`.

2. **Replay buffer size:** Paper uses 1M transitions. We use 200k due to RAM constraints (1M would require ~56 GB).

3. **Epsilon decay mismatch:** Our epsilon decays over 1M agent steps (= 4M environment frames). The paper decays over 1M environment frames (= 250k agent steps). Our agent explores randomly for 4x longer.

4. **Optimizer:** Paper uses RMSProp (lr=0.00025, α=0.95, ε=0.01). We use Adam (lr=1e-4).

## Environment Versions

- Python: 3.11
- PyTorch: 2.11.0+cu128
- Gymnasium: 1.2.3
- ALE-py: 0.11.2

## Compute

| Run | Device | Steps | Approx Time |
|-----|--------|-------|-------------|
| Seed 0 (2M) | Apple M4 MPS | 2M | ~3 hours |
| Seed 0 (10M) | NVIDIA RTX PRO 500 Blackwell | 10M | ~17 hours |
| Seed 1 (10M) | NVIDIA RTX PRO 500 Blackwell | 10M | ~17 hours |
| Seed 2 (10M) | NVIDIA RTX PRO 500 Blackwell | 10M | ~17 hours |

## References

- Mnih et al. (2015). *Human-level control through deep reinforcement learning*. Nature.
- [ALE / Gymnasium](https://github.com/Farama-Foundation/Arcade-Learning-Environment)
