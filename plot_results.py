"""
Plot DQN training results from TensorBoard event files.
Shows mean +/- std across seeds 0, 1, 2 for the 10M run.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def read_tfevents(path):
    scalars = {}
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    while offset < len(data):
        if offset + 12 > len(data):
            break
        length = struct.unpack_from("<Q", data, offset)[0]
        offset += 12
        if offset + length + 4 > len(data):
            break
        record = data[offset: offset + length]
        offset += length + 4
        try:
            _parse_event(record, scalars)
        except Exception:
            pass
    return scalars


def _read_varint(data, pos):
    result, shift = 0, 0
    while True:
        b = data[pos]; pos += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            return result, pos


def _parse_proto(data, start, end):
    fields = {}
    pos = start
    while pos < end:
        tag, pos = _read_varint(data, pos)
        field_num, wire_type = tag >> 3, tag & 7
        if wire_type == 0:
            val, pos = _read_varint(data, pos)
            fields.setdefault(field_num, []).append(val)
        elif wire_type == 1:
            fields.setdefault(field_num, []).append(data[pos:pos+8]); pos += 8
        elif wire_type == 2:
            length, pos = _read_varint(data, pos)
            fields.setdefault(field_num, []).append(data[pos:pos+length]); pos += length
        elif wire_type == 5:
            fields.setdefault(field_num, []).append(data[pos:pos+4]); pos += 4
        else:
            break
    return fields


def _parse_event(record, scalars):
    event = _parse_proto(record, 0, len(record))
    step = event[2][0] if 2 in event else 0
    if 5 not in event:
        return
    for summary_bytes in event[5]:
        summary = _parse_proto(summary_bytes, 0, len(summary_bytes))
        if 1 not in summary:
            continue
        for value_bytes in summary[1]:
            value = _parse_proto(value_bytes, 0, len(value_bytes))
            if 1 not in value or 2 not in value:
                continue
            tag = value[1][0].decode("utf-8")
            float_val = struct.unpack_from("<f", value[2][0])[0]
            scalars.setdefault(tag, []).append((step, float_val))


def smooth(values, weight=0.7):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return np.array(smoothed)


def get_scalars(run_id, tag):
    path = os.path.join("results", "runs", run_id)
    tf_file = next(f for f in os.listdir(path) if f.startswith("events"))
    scalars = read_tfevents(os.path.join(path, tf_file))
    if tag not in scalars:
        return np.array([]), np.array([])
    pts = sorted(scalars[tag])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


# ── runs ──────────────────────────────────────────────────────────────────────
runs_dir = "results/runs"
seed_runs = {
    0: "Breakout-v5_seed0_1776070538",
    1: "Breakout-v5_seed1_1776135513",
    2: "Breakout-v5_seed2_1776194883",
}
COLORS = {"mean": "#2196F3", "seeds": ["#4CAF50", "#FF9800", "#9C27B0"]}

# ── collect eval data across seeds ───────────────────────────────────────────
eval_data = {}
for seed, run_id in seed_runs.items():
    steps, vals = get_scalars(run_id, "eval/mean_reward")
    eval_data[seed] = (steps, vals)

# Align to common steps (trim to shortest seed)
min_len = min(len(eval_data[s][0]) for s in [0, 1, 2])
ref_steps = eval_data[0][0][:min_len]
aligned = np.stack([eval_data[s][1][:min_len] for s in [0, 1, 2]])
print(f"Eval points per seed: { {s: len(eval_data[s][0]) for s in [0,1,2]} }")
mean_eval = aligned.mean(axis=0)
std_eval  = aligned.std(axis=0)

# ── collect train avg100 across seeds ─────────────────────────────────────────
train_data = {}
for seed, run_id in seed_runs.items():
    steps, vals = get_scalars(run_id, "train/avg_reward_100")
    train_data[seed] = (steps, vals)

min_train_len = min(len(train_data[s][0]) for s in [0, 1, 2])
ref_train_steps = train_data[0][0][:min_train_len]
aligned_train = np.stack([train_data[s][1][:min_train_len] for s in [0, 1, 2]])
mean_train = aligned_train.mean(axis=0)
std_train  = aligned_train.std(axis=0)

# ── collect fps ───────────────────────────────────────────────────────────────
fps_per_seed = {}
for seed, run_id in seed_runs.items():
    _, vals = get_scalars(run_id, "train/fps")
    fps_per_seed[seed] = vals

# ── plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("DQN on ALE/Breakout-v5 — 10M Steps, Seeds 0/1/2", fontsize=15, fontweight="bold", y=0.98)
gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])   # top: eval mean across seeds (full width)
ax2 = fig.add_subplot(gs[1, 0])   # bottom left: individual seeds
ax3 = fig.add_subplot(gs[1, 1])   # bottom right: train avg100

# ── ax1: mean ± std eval ──────────────────────────────────────────────────────
x = ref_steps / 1e6
sm_mean = smooth(mean_eval, weight=0.6)
ax1.plot(x, sm_mean, color=COLORS["mean"], linewidth=2.2, label="Mean (seeds 0–2)")
ax1.fill_between(x, mean_eval - std_eval, mean_eval + std_eval,
                 alpha=0.2, color=COLORS["mean"], label="±1 std")
ax1.set_title("Eval Mean Reward — Mean ± Std across 3 Seeds", fontsize=12)
ax1.set_xlabel("Timesteps (M)")
ax1.set_ylabel("Mean Reward (unclipped)")
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 10)

# ── ax2: individual seed eval curves ─────────────────────────────────────────
for seed, color in zip([0, 1, 2], COLORS["seeds"]):
    steps, vals = eval_data[seed]
    sm = smooth(vals, weight=0.6)
    ax2.plot(steps / 1e6, sm, color=color, linewidth=1.6, label=f"Seed {seed}")
ax2.set_title("Eval Reward — Individual Seeds", fontsize=12)
ax2.set_xlabel("Timesteps (M)")
ax2.set_ylabel("Mean Reward (unclipped)")
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 10)

# ── ax3: train avg100 mean ± std ──────────────────────────────────────────────
x_train = ref_train_steps / 1e6
sm_train = smooth(mean_train, weight=0.9)
ax3.plot(x_train, sm_train, color=COLORS["mean"], linewidth=2.0, label="Mean (seeds 0–2)")
ax3.fill_between(x_train, mean_train - std_train, mean_train + std_train,
                 alpha=0.2, color=COLORS["mean"], label="±1 std")
ax3.set_title("Train Avg Reward (100-ep rolling, clipped)", fontsize=12)
ax3.set_xlabel("Timesteps (M)")
ax3.set_ylabel("Avg Reward (clipped)")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xlim(0, 10)

out = "results/training_3seeds.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")

# Print final stats for README
print("\n=== Final Stats ===")
for s in [0, 1, 2]:
    steps, vals = eval_data[s]
    print(f"Seed {s}: final_eval={vals[-1]:.1f}, max_eval={max(vals):.1f}")
plt.show()
