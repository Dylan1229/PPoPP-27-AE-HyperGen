"""Figure 12: Stage-2 latency of CogVideoX at 4K as HyperGen's mechanisms are added one at a time:
1 GPU, 4-GPU tile parallelism, the cache with a uniform threshold, region-aware cache control, and
cache-guided workload rebalancing.

Reads results/T7_figures/timings.tsv (block fig12) and writes figures/fig12_breakdown.pdf. Every
mechanism acts on Stage 2, while Stage 1 and the upsampling are identical in all bars, so the figure
plots Stage 2. The `+Cache` and `+Region-aware` steps differ only in the per-tile scale factor (0,
i.e. a uniform threshold, vs. 0.5).

    RESULTS_DIR=results/paper python plots/fig12_breakdown.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rice_palette import sfp, RICE  # noqa: E402

HERE = Path(__file__).resolve().parent
from _paths import RES, TABLES, FIGS, PREV  # noqa: E402  (RESULTS_DIR / OUT_DIR)
FIG = FIGS
TIME = RES / "T7_figures/timings.tsv"
S1 = RES / "T8_stage1/timings.tsv"

C_OURS = RICE["ours"]

# The cascade, in order, with the paper's labels.
STEPS = [
    ("baseline",    "Baseline"),
    ("tilepar",     "+ Tile Parallelism\n(4 GPUs)"),
    ("cache",       "+ Cache"),
    ("regionaware", "+ Region-aware\nCache Control"),
    ("rebalance",   "+ Workload Rebalance"),
]


def stage1_for(cfg: str) -> float:
    """Stage 1, measured once on an exclusive host with the latent generated rather than loaded."""
    if not S1.exists():
        raise SystemExit(f"missing {S1} -- Stage 1 must be measured for an end-to-end figure")
    with open(S1) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["config"] == cfg and r["rc"] == "0" and r.get("stage1_s"):
                return float(r["stage1_s"])
    return 0.0   # only used for the printed note on why Stage 1 is excluded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="cog")
    ap.add_argument("--res", default="4k")
    ap.add_argument("--name", default="fig12_breakdown")
    a = ap.parse_args()

    if not TIME.exists():
        sys.exit(f"missing {TIME}")
    rows = {}
    with open(TIME) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["block"] == "fig12" and r["config"] == a.config and r["res"] == a.res \
               and r["rc"] == "0":
                rows[r["variant"]] = r

    s1 = stage1_for(a.config) if S1.exists() else 0.0
    lat, labels, detail = [], [], []
    for key, label in STEPS:
        r = rows.get(key)
        if not r or not r.get("stage2_s"):
            print(f"note: fig12 {a.config} {a.res} '{key}' not measured (a --quick run skips the 1-GPU "
                  f"baseline); plotting the remaining steps", file=sys.stderr)
            continue
        up = float(r["upscale_s"] or 0)
        s2 = float(r["stage2_s"])
        lat.append(s2)
        labels.append(label)
        detail.append((label.replace("\n", " "), s1, up, s2, r.get("skip_rate") or "-"))

    print(f"Stage-2 latency, {a.config} at {a.res}")
    print(f"  (Stage 1 = {s1:.1f}s and upscale ~{detail[0][2]:.0f}s are identical in every bar and "
          f"not plotted)")
    for lab, _s1, up, s2, sk in detail:
        print(f"  {lab:<32} stage2={s2:7.1f}s   skip={sk}")
    total = lat[0] / lat[-1]
    print(f"  total speedup {lat[0]:.0f}s -> {lat[-1]:.0f}s = {total:.2f}x")

    sfp.apply_publication_style(sfp.FigureStyle(font_size=17, axes_linewidth=1.6))
    fig, ax = plt.subplots(figsize=(11, 3.7))
    sp, lp = labels[::-1], lat[::-1]
    y = np.arange(len(sp))
    ax.barh(y, lp, height=0.6, color=C_OURS, edgecolor="black", lw=0.8, alpha=0.95)
    for yi, v in zip(y, lp):
        ax.annotate(f"{int(round(v))}", (v, yi), xytext=(-7, 0), textcoords="offset points",
                    va="center", ha="right", color="white", fontsize=13, fontweight="bold")
    ax.set_yticks(y)
    ax.set_yticklabels(sp, fontsize=15, fontweight="bold")
    ax.set_xlabel("Latency (s)", fontweight="bold", loc="right")
    ax.set_xlim(0, max(lat) * 1.22)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", ls="--", alpha=0.4)
    ax.set_axisbelow(True)
    ax.axvline(lat[0], color="gray", ls="--", lw=1.3, alpha=0.5)

    # Per-step speedup arrows, cascading down the bars.
    for i in range(len(lat) - 1):
        ci, ni = len(lat) - 1 - i, len(lat) - 2 - i
        cur, nxt = lp[ci], lp[ni]
        if nxt <= 0:
            continue
        ax.annotate("", xy=(nxt, ni), xytext=(cur, ci),
                    arrowprops=dict(arrowstyle="->", color=RICE["ablation"], lw=1.6,
                                    connectionstyle="arc3,rad=-0.18"))
        ax.annotate(f"{cur / nxt:.2f}$\\times$", ((cur + nxt) / 2, (ci + ni) / 2),
                    xytext=(14, 0), textcoords="offset points", va="center",
                    color=RICE["ablation"], fontsize=12, fontweight="bold")
    # Below the top bar, not on it: at 4K the baseline bar reaches ~97% of the axis and the label
    # landed on top of its own value.
    ax.annotate(f"total {total:.2f}$\\times$", (max(lat) * 1.02, len(sp) - 1.42),
                ha="left", va="center", fontsize=14, fontweight="bold")

    sfp.finalize_figure(fig, FIG / f"{a.name}.pdf", formats=["pdf"], close=False, pad=0.04)
    sfp.finalize_figure(fig, PREV / f"{a.name}.png", formats=["png"], dpi=110, close=True, pad=0.04)
    print(f"\nwrote {FIG / a.name}.pdf")


if __name__ == "__main__":
    main()
