"""
Plot the Phase-2 ablation on MiniGrid-MemoryS7-v0:
    Baseline DQN  vs  DRQN (CNN+LSTM, L=10)  vs  Frame-stacking DQN (K=10, no LSTM)

Isolates recurrence as the design change: framestack has the same K=10 context
window as DRQN's BPTT length but no hidden state carrying info across windows.
"""

import os
import struct
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── tfevents reader (same protocol as plot_results_minigrid.py) ───────────────
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


def get_scalars(run_id, tag):
    path = os.path.join("results", "runs", run_id)
    tf_file = next(f for f in os.listdir(path) if f.startswith("events"))
    scalars = read_tfevents(os.path.join(path, tf_file))
    if tag not in scalars:
        return np.array([]), np.array([])
    pts = sorted(scalars[tag])
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def discover_runs(prefix):
    runs_dir = "results/runs"
    seed_runs = {}
    for folder in sorted(os.listdir(runs_dir)):
        if folder.startswith(prefix):
            seed = int(folder.split("seed")[1].split("_")[0])
            if seed not in seed_runs or folder > seed_runs[seed]:
                seed_runs[seed] = folder
    return seed_runs


def collect_data(seed_runs, tag):
    return {seed: get_scalars(run_id, tag) for seed, run_id in seed_runs.items()}


def mean_std(data, seeds):
    min_len = min(len(data[s][1]) for s in seeds)
    ref_steps = data[seeds[0]][0][:min_len]
    stacked = np.stack([data[s][1][:min_len] for s in seeds])
    return ref_steps, stacked.mean(axis=0), stacked.std(axis=0)


# ── discover runs ─────────────────────────────────────────────────────────────
baseline_runs   = discover_runs("memory_baseline_seed")
drqn_runs       = discover_runs("memory_drqn_seed")
framestack_runs = discover_runs("memory_framestack_seed")

print(f"Baseline runs:   {baseline_runs}")
print(f"DRQN runs:       {drqn_runs}")
print(f"Framestack runs: {framestack_runs}")

assert len(baseline_runs)   >= 1, "No baseline runs found"
assert len(drqn_runs)       >= 1, "No DRQN runs found"
assert len(framestack_runs) >= 1, "No framestack runs found"

baseline_seeds   = sorted(baseline_runs)
drqn_seeds       = sorted(drqn_runs)
framestack_seeds = sorted(framestack_runs)

# ── collect metrics ───────────────────────────────────────────────────────────
bl_sr = collect_data(baseline_runs,   "eval/success_rate")
bl_rw = collect_data(baseline_runs,   "eval/mean_reward")
dr_sr = collect_data(drqn_runs,       "eval/success_rate")
dr_rw = collect_data(drqn_runs,       "eval/mean_reward")
fs_sr = collect_data(framestack_runs, "eval/success_rate")
fs_rw = collect_data(framestack_runs, "eval/mean_reward")

bl_steps_sr, bl_mean_sr, bl_std_sr = mean_std(bl_sr, baseline_seeds)
bl_steps_rw, bl_mean_rw, bl_std_rw = mean_std(bl_rw, baseline_seeds)
dr_steps_sr, dr_mean_sr, dr_std_sr = mean_std(dr_sr, drqn_seeds)
dr_steps_rw, dr_mean_rw, dr_std_rw = mean_std(dr_rw, drqn_seeds)
fs_steps_sr, fs_mean_sr, fs_std_sr = mean_std(fs_sr, framestack_seeds)
fs_steps_rw, fs_mean_rw, fs_std_rw = mean_std(fs_rw, framestack_seeds)

# ── plot ──────────────────────────────────────────────────────────────────────
BL_COLOR = "#FF7043"   # orange-red
DR_COLOR = "#2196F3"   # blue
FS_COLOR = "#7E57C2"   # purple — ablation
SEED_COLORS = ["#4CAF50", "#FF9800", "#9C27B0"]

fig = plt.figure(figsize=(16, 11))
fig.suptitle(
    "Phase-2 Ablation on MiniGrid-MemoryS7-v0: does recurrence matter?",
    fontsize=15, fontweight="bold", y=0.99,
)
gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

# ── ax1: success rate headline comparison ─────────────────────────────────────
bl_label = f"Baseline DQN (n={len(baseline_seeds)})"
dr_label = f"DRQN / CNN+LSTM, L=10 (n={len(drqn_seeds)})"
fs_label = f"Frame-stack DQN, K=10 (n={len(framestack_seeds)})"

def plot_band(ax, steps, mean, std, color, label, linewidth=2.2, scale=100):
    ax.plot(steps / 1e6, mean * scale, color=color, linewidth=linewidth, label=label)
    ax.fill_between(steps / 1e6,
                    (mean - std) * scale,
                    (mean + std) * scale,
                    alpha=0.2, color=color)

plot_band(ax1, bl_steps_sr, bl_mean_sr, bl_std_sr, BL_COLOR, bl_label)
plot_band(ax1, dr_steps_sr, dr_mean_sr, dr_std_sr, DR_COLOR, dr_label)
plot_band(ax1, fs_steps_sr, fs_mean_sr, fs_std_sr, FS_COLOR, fs_label)

ax1.axhline(50, color="gray", linestyle="--", linewidth=1.6, label="Random chance (50%)")
ax1.set_title("Eval Success Rate (mean ± 1 std across seeds)", fontsize=12)
ax1.set_xlabel("Training steps (millions)")
ax1.set_ylabel("Success rate (%)")
ax1.legend(fontsize=10, loc="upper left")
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 105)

# ── ax2: framestack individual seeds vs DRQN mean ─────────────────────────────
for seed, color in zip(framestack_seeds, SEED_COLORS):
    steps, vals = fs_sr[seed]
    ax2.plot(steps / 1e6, vals * 100, color=color,
             linewidth=1.6, label=f"Framestack seed {seed}", alpha=0.9)
ax2.plot(dr_steps_sr / 1e6, dr_mean_sr * 100,
         color=DR_COLOR, linewidth=2.0, linestyle="--",
         label="DRQN mean (for reference)")
ax2.axhline(50, color="gray", linestyle="--", linewidth=1.4, label="Random chance")
ax2.set_title("Framestack per-seed success rate", fontsize=12)
ax2.set_xlabel("Training steps (millions)")
ax2.set_ylabel("Success rate (%)")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 105)

# ── ax3: mean reward comparison ───────────────────────────────────────────────
plot_band(ax3, bl_steps_rw, bl_mean_rw, bl_std_rw, BL_COLOR, bl_label, linewidth=2.0, scale=1)
plot_band(ax3, dr_steps_rw, dr_mean_rw, dr_std_rw, DR_COLOR, dr_label, linewidth=2.0, scale=1)
plot_band(ax3, fs_steps_rw, fs_mean_rw, fs_std_rw, FS_COLOR, fs_label, linewidth=2.0, scale=1)
ax3.set_title("Eval Mean Reward", fontsize=12)
ax3.set_xlabel("Training steps (millions)")
ax3.set_ylabel("Mean reward")
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.3)

out = "results/minigrid_ablation_framestack.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")

# ── summary stats ─────────────────────────────────────────────────────────────
def summarize(name, seeds, sr, rw):
    print(f"\n=== {name} ===")
    for seed in seeds:
        _, s = sr[seed]
        _, r = rw[seed]
        print(f"  Seed {seed}: final_success={s[-1]*100:5.1f}%  "
              f"best_success={s.max()*100:5.1f}%  final_reward={r[-1]:.3f}")

summarize("Baseline DQN",      baseline_seeds,   bl_sr, bl_rw)
summarize("DRQN (CNN+LSTM)",   drqn_seeds,       dr_sr, dr_rw)
summarize("Frame-stack (K=10)", framestack_seeds, fs_sr, fs_rw)
