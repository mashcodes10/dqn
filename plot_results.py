"""
Plot DQN training results from TensorBoard event files.
"""

import os
import struct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def read_tfevents(path):
    """Parse a TFEvents file and return {tag: [(step, value), ...]}."""
    scalars = {}
    with open(path, "rb") as f:
        data = f.read()

    offset = 0
    while offset < len(data):
        if offset + 12 > len(data):
            break
        length = struct.unpack_from("<Q", data, offset)[0]
        offset += 12  # uint64 length + uint32 masked_crc32
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


def smooth(values, weight=0.6):
    smoothed, last = [], values[0]
    for v in values:
        last = last * weight + v * (1 - weight)
        smoothed.append(last)
    return smoothed


# ── locate runs ──────────────────────────────────────────────────────────────
runs_dir = os.path.join("results", "runs")
runs = {
    "2M (Mac MPS)":  "Breakout-v5_seed0_1775988250",
    "5M (CUDA)":     "Breakout-v5_seed0_1776032526",
    "10M (CUDA)":    "Breakout-v5_seed0_1776070538",
}
colors = {"2M (Mac MPS)": "#4C72B0", "5M (CUDA)": "#DD8452", "10M (CUDA)": "#55A868"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("DQN on ALE/Breakout-v5 — Seed 0", fontsize=14, fontweight="bold")

for label, run_id in runs.items():
    path = os.path.join(runs_dir, run_id)
    tf_file = next(f for f in os.listdir(path) if f.startswith("events"))
    scalars = read_tfevents(os.path.join(path, tf_file))
    color = colors[label]

    # ── Eval mean reward ─────────────────────────────────────────────────────
    if "eval/mean_reward" in scalars:
        pts = sorted(scalars["eval/mean_reward"])
        steps = [p[0] / 1e6 for p in pts]
        means = [p[1] for p in pts]
        stds  = [p[1] for p in sorted(scalars.get("eval/std_reward", []))]

        sm = smooth(means)
        axes[0].plot(steps, sm, color=color, label=label, linewidth=1.8)
        axes[0].fill_between(
            steps,
            [m - s for m, s in zip(means, stds)],
            [m + s for m, s in zip(means, stds)],
            alpha=0.12, color=color
        )

    # ── Train avg100 reward ───────────────────────────────────────────────────
    if "train/avg_reward_100" in scalars:
        pts = sorted(scalars["train/avg_reward_100"])
        steps = [p[0] / 1e6 for p in pts]
        vals  = [p[1] for p in pts]
        sm = smooth(vals, weight=0.85)
        axes[1].plot(steps, sm, color=color, label=label, linewidth=1.8)

# ── axes formatting ───────────────────────────────────────────────────────────
axes[0].set_title("Eval Mean Reward (unclipped, full games)")
axes[0].set_xlabel("Timesteps (M)")
axes[0].set_ylabel("Mean Reward")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_xlim(left=0)

axes[1].set_title("Train Avg Reward (100-ep rolling, clipped)")
axes[1].set_xlabel("Timesteps (M)")
axes[1].set_ylabel("Avg Reward (clipped)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(left=0)

plt.tight_layout()
out = "results/training_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.show()
