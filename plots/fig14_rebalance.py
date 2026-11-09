"""Figure 14: Stage-2 latency at 4K with the cache on, without and with cache-guided rebalancing.

Reads the block=fig14 rows of $RESULTS_DIR/T7_figures/timings.tsv (scripts/run_fig14.sh; variants
cache_reb0 / cache_reb1) and draws one panel per backbone, annotated with the rebalancing speedup.
--paper plots the paper's values instead (results/paper/T7_figures/fig13_fig14.tsv).

    RESULTS_DIR=results/ae    python plots/fig14_rebalance.py
    RESULTS_DIR=results/paper python plots/fig14_rebalance.py --paper
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
from _paths import RES, FIGS, PREV  # noqa: E402

TITLES = {"cog": "CogVideo (4K)", "hunyuan": "HunyuanVideo (4K)"}


def load(paper: bool):
    path = RES / "T7_figures" / ("fig13_fig14.tsv" if paper else "timings.tsv")
    if not path.exists():
        sys.exit(f"missing {path} -- run scripts/run_fig14.sh")
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["block"] != "fig14" or not r.get("stage2_s") or r["res"] != "4k":
                continue
            if not paper and r.get("rc") != "0":
                continue
            data.setdefault(r["config"], {}).setdefault(int(r["ngpu"]), {})[r["variant"]] = float(r["stage2_s"])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="plot the paper's values (T7_figures/fig13_fig14.tsv)")
    ap.add_argument("--name", default="fig14_rebalance")
    a = ap.parse_args()
    data = load(a.paper)
    # Keep only GPU counts that have both arms.
    data = {c: {n: v for n, v in d.items() if {"cache_reb0", "cache_reb1"} <= set(v)} for c, d in data.items()}
    cfgs = [c for c in ("cog", "hunyuan") if data.get(c)]
    if not cfgs:
        sys.exit("no complete fig14 pairs (w/o and w/ rebalance) to plot")

    print(f"Figure 14, Stage-2 latency (s), 4K, cache on{' [paper]' if a.paper else ''}:")
    sfp.apply_publication_style(sfp.FigureStyle(font_size=18, axes_linewidth=1.6))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 3.2), squeeze=False)
    bw = 0.36
    for ax, cfg in zip(axes[0], cfgs):
        g = sorted(data[cfg]); x = np.arange(len(g))
        va = [data[cfg][k]["cache_reb0"] for k in g]
        vo = [data[cfg][k]["cache_reb1"] for k in g]
        ax.bar(x - bw / 2, va, bw, color=RICE["other_baseline"], edgecolor="black", lw=0.8, label="w/o rebalance")
        ax.bar(x + bw / 2, vo, bw, color=RICE["ours"], edgecolor="black", lw=0.8, label="w/ rebalance")
        for xi, (pa, po) in enumerate(zip(va, vo)):
            print(f"  {cfg:8s} {g[xi]} GPUs: {pa:7.1f}s -> {po:7.1f}s  ({pa / po:.2f}x)")
            ax.annotate(f"{pa:.0f}", (x[xi] - bw / 2, pa * 0.5), ha="center", va="center", color="white",
                        fontsize=12, fontweight="bold")
            ax.annotate(f"{po:.0f}", (x[xi] + bw / 2, po * 0.5), ha="center", va="center", color="white",
                        fontsize=12, fontweight="bold")
            ax.annotate(f"{pa / po:.2f}$\\times$", (x[xi] + bw / 2, pa), xytext=(5, 3),
                        textcoords="offset points", ha="left", va="bottom", fontsize=14,
                        fontweight="bold", color=RICE["ablation"])
        ax.set_xticks(x); ax.set_xticklabels([f"{k} GPUs" for k in g], fontweight="bold")
        ax.set_title(TITLES.get(cfg, cfg), fontweight="bold")
        ax.set_ylim(0, max(va) * 1.6)
        ax.legend(loc="upper center", ncol=2, fontsize=13, handlelength=1.2, columnspacing=1.0)
        ax.grid(axis="y", ls="--", alpha=0.35); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Latency (s)", fontweight="bold")
    fig.subplots_adjust(wspace=0.22)
    sfp.finalize_figure(fig, FIGS / f"{a.name}.pdf", formats=["pdf"], close=False, pad=0.04)
    sfp.finalize_figure(fig, PREV / f"{a.name}.png", formats=["png"], dpi=110, close=True, pad=0.04)
    print(f"wrote {FIGS / a.name}.pdf")


if __name__ == "__main__":
    main()
