"""Figure 11: end-to-end latency (Stage 1 + upsampling + Stage 2) in three settings: Origin (1 GPU,
no cache), +Parallelism (4 GPUs at 2K, 8 at 4K, workload rebalance on), and +Parallelism +Cache.

Reads results/T7_figures/timings.tsv (block fig10) and results/T8_stage1/timings.tsv, and writes
figures/fig11_e2e.pdf. Stage 1 runs on one GPU in every setting, so it is measured once per
backbone (T8_stage1) and added to every bar; the bars' own runs load the shared Stage-1 sketch.

    RESULTS_DIR=results/paper python plots/fig11_e2e.py
    RESULTS_DIR=results/ae    python plots/fig11_e2e.py --configs cog
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

TITLES = {"cog": "CogVideo", "hunyuan": "HunyuanVideo",
          "wan_i2v": "Wan2.1 I2V", "wan_t2v": "Wan2.1 T2V"}
CATS = [("origin", "Origin"), ("parallel", "+Parallelism"), ("parcache", "+Parallelism +Cache")]
# other_baseline grey -> best_baseline salmon -> ours blue, i.e. worst to best left to right.
COLORS = [RICE["other_baseline"], RICE["best_baseline"], RICE["ours"]]
RES = ["2K", "4K"]


def load_stage1():
    out = {}
    if not S1.exists():
        return out
    with open(S1) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["rc"] == "0" and r.get("stage1_s"):
                # Last wins: a later measurement supersedes an earlier one.
                out[r["config"]] = float(r["stage1_s"])
    return out


def load_fig10():
    if not TIME.exists():
        sys.exit(f"missing {TIME}")
    out = {}
    with open(TIME) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["block"] == "fig10" and r["rc"] == "0" and r.get("stage2_s"):
                out[(r["config"], r["res"], r["variant"])] = r
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="*", default=None,
                    help="default: every backbone with a complete set of six points")
    ap.add_argument("--name", default="fig11_e2e")
    a = ap.parse_args()

    rows, s1s = load_fig10(), load_stage1()

    # Only backbones with all six points AND a Stage-1 measurement. A partial panel would invite the
    # reader to compare a 2K bar against a missing 4K one.
    complete, skipped, res_of = [], [], {}
    for cfg in (a.configs or ["cog", "hunyuan", "wan_i2v", "wan_t2v"]):
        if not a.configs and not any(k[0] == cfg for k in rows):
            continue   # backbone not measured at all: nothing to report
        have = [r for r in RES if all((cfg, r.lower(), v) in rows for v, _ in CATS)]
        if have and cfg in s1s:
            complete.append(cfg)
            res_of[cfg] = have
            if len(have) < len(RES):
                print(f"note: {cfg} has only {', '.join(have)} (a --quick run); plotting what exists",
                      file=sys.stderr)
        else:
            skipped.append((cfg, "no Stage-1 measurement" if have else "incomplete fig10 points"))
    if skipped:
        print("skipped (data incomplete):", file=sys.stderr)
        for cfg, why in skipped:
            print(f"  {cfg}: {why}", file=sys.stderr)
    if not complete:
        sys.exit("no backbone has a complete set of six fig10 points plus a Stage-1 measurement")

    data = {}
    for cfg in complete:
        s1 = s1s[cfg]
        data[cfg] = {}
        print(f"{TITLES[cfg]} (Stage 1 = {s1:.1f}s, constant across settings):")
        for res in res_of[cfg]:
            vals = []
            for v, label in CATS:
                r = rows[(cfg, res.lower(), v)]
                up = float(r["upscale_s"] or 0)
                s2 = float(r["stage2_s"])
                vals.append(s1 + up + s2)
                print(f"  {res} {label:<21} {s1:6.1f} + {up:6.1f} + {s2:7.1f} = {vals[-1]:7.1f}s"
                      f"   ({r['ngpu']} GPU)")
            data[cfg][res] = vals
            print(f"  {res} speedup {vals[0] / vals[-1]:.2f}x")

    sfp.apply_publication_style(sfp.FigureStyle(font_size=18, axes_linewidth=1.6))
    fig, axes = plt.subplots(1, len(complete), figsize=(6 * len(complete), 3.2), squeeze=False)
    bw = 0.24
    for ax, cfg in zip(axes[0], complete):
        d = data[cfg]
        RS = res_of[cfg]
        gx = np.arange(len(RS)) * 1.0
        for i, (v, label) in enumerate(CATS):
            pos = gx + (i - 1) * bw
            heights = [d[r][i] for r in RS]
            ax.bar(pos, heights, bw, color=COLORS[i], edgecolor="black", lw=0.8, alpha=0.95,
                   label=label if cfg == complete[0] else None)
            for x, val in zip(pos, heights):
                ax.annotate(f"{int(round(val))}", (x, val),
                            xytext=(6 if i == 2 else 0, 4 if i == 2 else 2),
                            textcoords="offset points",
                            ha="left" if i == 2 else "center", va="bottom",
                            fontsize=12, fontweight="bold")
        ymax = max(max(v) for v in d.values())
        # Speedup arc from Origin to the full system.
        for i, r in enumerate(RS):
            x0, y0 = gx[i] - bw, d[r][0]
            x2, y2 = gx[i] + bw, d[r][2]
            cx, cy = (x0 + x2) / 2, y0 + ymax * 0.10
            t = np.linspace(0, 1, 60)
            bx = (1 - t) ** 2 * x0 + 2 * (1 - t) * t * cx + t ** 2 * x2
            by = (1 - t) ** 2 * y0 + 2 * (1 - t) * t * cy + t ** 2 * y2
            ax.plot(bx, by, color=RICE["ablation"], lw=1.5, alpha=0.85)
            ax.annotate(f"{d[r][0] / d[r][2]:.2f}$\\times$", (cx, cy),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        color=RICE["ablation"], fontsize=13, fontweight="bold")
        ax.set_xticks(gx)
        ax.set_xticklabels(RS, fontweight="bold")
        ax.set_title(TITLES[cfg], fontweight="bold")
        ax.set_ylim(0, ymax * 1.30)
        ax.grid(axis="y", ls="--", alpha=0.4)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("Latency (s)", fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.10), fontsize=15)

    sfp.finalize_figure(fig, FIG / f"{a.name}.pdf", formats=["pdf"], close=False, pad=0.04)
    sfp.finalize_figure(fig, PREV / f"{a.name}.png", formats=["png"], dpi=110, close=True, pad=0.04)
    print(f"\nwrote {FIG / a.name}.pdf  ({len(complete)} panel(s): {', '.join(complete)})")


if __name__ == "__main__":
    main()
