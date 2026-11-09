"""Figure 13: tile-parallel scalability of Stage 2 (no cache, no rebalance).

Reads the block=fig13 rows of $RESULTS_DIR/T7_figures/timings.tsv (scripts/run_fig13.sh) and plots
Stage-2 latency against the GPU count, one panel per backbone, annotated with the speedup over the
same backbone and resolution on 1 GPU. --paper plots the paper's values instead
(results/paper/T7_figures/fig13_fig14.tsv).

    RESULTS_DIR=results/ae    python plots/fig13_scalability.py
    RESULTS_DIR=results/paper python plots/fig13_scalability.py --paper
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rice_palette import sfp, RICE  # noqa: E402
from _paths import RES, FIGS, PREV  # noqa: E402

TITLES = {"cog": "CogVideo", "hunyuan": "HunyuanVideo", "wan_i2v": "Wan2.1 I2V", "wan_t2v": "Wan2.1 T2V"}
STYLE = {"2k": dict(color=RICE["ours"], marker="o", ls="-"),
         "4k": dict(color=RICE["ablation"], marker="s", ls="--")}


def load(paper: bool):
    path = RES / "T7_figures" / ("fig13_fig14.tsv" if paper else "timings.tsv")
    if not path.exists():
        sys.exit(f"missing {path} -- run scripts/run_fig13.sh")
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["block"] != "fig13" or not r.get("stage2_s"):
                continue
            if not paper and r.get("rc") != "0":
                continue
            data.setdefault(r["config"], {}).setdefault(r["res"], {})[int(r["ngpu"])] = float(r["stage2_s"])
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", action="store_true", help="plot the paper's values (T7_figures/fig13_fig14.tsv)")
    ap.add_argument("--name", default="fig13_scalability")
    a = ap.parse_args()
    data = load(a.paper)
    cfgs = [c for c in ("cog", "hunyuan", "wan_i2v", "wan_t2v") if c in data]
    if not cfgs:
        sys.exit("no fig13 rows to plot")

    print(f"Figure 13, Stage-2 latency (s) and speedup over 1 GPU{' [paper]' if a.paper else ''}:")
    sfp.apply_publication_style(sfp.FigureStyle(font_size=18, axes_linewidth=1.6))
    fig, axes = plt.subplots(1, len(cfgs), figsize=(6 * len(cfgs), 3.2), squeeze=False)
    for ax, cfg in zip(axes[0], cfgs):
        for res in ("2k", "4k"):
            pts = data[cfg].get(res)
            if not pts:
                continue
            g = sorted(pts)
            v = [pts[k] for k in g]
            ax.plot(g, v, lw=2.6, markersize=8, label=res.upper(), **STYLE[res])
            base = pts.get(1)
            for k, val in zip(g, v):
                sp = f"{base / val:.1f}x" if base else ""
                print(f"  {cfg:8s} {res} {k} GPU: {val:7.1f}s  {sp}")
                if base:
                    ax.annotate(f"{base / val:.1f}$\\times$", (k, val), xytext=(4, 8),
                                textcoords="offset points", fontsize=13, fontweight="bold",
                                color=STYLE[res]["color"])
        ax.set_xlabel("Number of GPUs", fontweight="bold")
        ax.set_title(TITLES[cfg], fontweight="bold")
        ax.set_xticks([1, 2, 4, 8]); ax.set_xlim(0.5, 8.7)
        ax.legend(loc="upper right")
        ax.grid(ls="--", alpha=0.3); ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Latency (s)", fontweight="bold")
    fig.subplots_adjust(wspace=0.22)
    sfp.finalize_figure(fig, FIGS / f"{a.name}.pdf", formats=["pdf"], close=False, pad=0.04)
    sfp.finalize_figure(fig, PREV / f"{a.name}.png", formats=["png"], dpi=110, close=True, pad=0.04)
    print(f"wrote {FIGS / a.name}.pdf")


if __name__ == "__main__":
    main()
