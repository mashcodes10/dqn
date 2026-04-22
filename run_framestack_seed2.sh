#!/usr/bin/env bash
# Mac / Linux launcher. Auto-detects CUDA -> MPS -> CPU.
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dqn
cd "$(dirname "$0")"
python train_minigrid_framestack.py --seed 2 --total-steps 2000000 --stack-k 10
