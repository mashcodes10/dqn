"""
Plot baseline DQN on MiniGrid MemoryEnv results.
Reuses plot_results.py structure — swaps metric to success_rate,
adds 50% random chance baseline, auto-discovers run folders.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# ── copy these functions exactly from plot_results.py (unchanged) ─────────────
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


# ── CHANGED: auto-discover memory baseline run folders ────────────────────────
runs_dir = "results/runs"
seed_runs = {}
for folder in sorted(os.listdir(runs_dir)):
    if folder.startswith("memory_baseline_seed"):
        # folder name: memory_baseline_seed0_<timestamp>
        seed = int(folder.split("seed")[1].split("_")[0])
        seed_runs[seed] = folder

print(f"Found runs: {seed_runs}")
assert len(seed_runs) >= 1, "No runs found"

COLORS = {"mean": "#2196F3", "seeds": ["#4CAF50", "#FF9800", "#9C27B0"]}

# ── CHANGED: collect success_rate and reward across seeds ─────────────────────
success_data = {}
reward_data  = {}
for seed, run_id in seed_runs.items():
    s_steps, s_vals = get_scalars(run_id, "eval/success_rate")   # ← new metric
    r_steps, r_vals = get_scalars(run_id, "eval/mean_reward")
    success_data[seed] = (s_steps, s_vals)
    reward_data[seed]  = (r_steps, r_vals)

# ── print stats for each seed ─────────────────────────────────────────────────
print("\n=== Stats ===")
for seed in sorted(seed_runs):
    steps, vals = success_data[seed]
    if len(vals) == 0:
        print(f"Seed {seed}: no data")
        continue
    print(f"Seed {seed}:")
    print(f"  Mean success rate: {np.mean(vals)*100:.1f}%")
    print(f"  Max success rate:  {np.max(vals)*100:.1f}%")
    print(f"  Final (last 10):   {np.mean(vals[-10:])*100:.1f}%")

# Align to shortest seed
min_len    = min(len(success_data[s][0]) for s in seed_runs)
ref_steps  = success_data[list(seed_runs.keys())[0]][0][:min_len]
aligned_sr = np.stack([success_data[s][1][:min_len] for s in sorted(seed_runs)])
aligned_rw = np.stack([reward_data[s][1][:min_len]  for s in sorted(seed_runs)])

mean_sr, std_sr = aligned_sr.mean(axis=0), aligned_sr.std(axis=0)
mean_rw, std_rw = aligned_rw.mean(axis=0), aligned_rw.std(axis=0)


fig, ax = plt.subplots(figsize=(12, 5))

seed = max(seed_runs, key=lambda s: seed_runs[s].split("_")[-1])
steps, vals = success_data[seed]
sm = smooth(vals * 100, weight=0.6)

ax.plot(steps, sm, color="#2196F3", linewidth=2.0, label=f"seed {seed}")
ax.axhline(50, color="red", linestyle="--", linewidth=1.5, label="Random chance (50%)")
ax.set_title("DQN on MiniGrid MemoryS7")
ax.set_xlabel("Timesteps")
ax.set_ylabel("Success Rate (%)")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(True, alpha=0.3)

out = f"results/memory_baseline_seed{seed}.png"
plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()