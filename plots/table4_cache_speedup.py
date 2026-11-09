"""Table 4: Stage-2 latency with and without the cache, the speedup, and the visual retention
(PSNR / SSIM / LPIPS of the cached video against the uncached one), per backbone, at 2K on 1 and 4
GPUs and at 4K on 4 GPUs.

Reads T7_figures/timings.tsv (block == fig11; rebalancing off), T7_figures/fig11_quality.tsv and
T7_figures/fig11_vbench.tsv, and writes tables/table4_cache_speedup.tex.

Latency is Stage 2 only: Stage 1 has no cache and the upsampling is unchanged. Each backbone runs at
its own cache threshold (scripts/config.sh).

    RESULTS_DIR=results/paper python plots/table4_cache_speedup.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _paths import RES, TABLES, FIGS, PREV  # noqa: E402  (RESULTS_DIR / OUT_DIR)
OUT = TABLES / "table4_cache_speedup.tex"
TIME = RES / "T7_figures/timings.tsv"
QUAL = RES / "T7_figures/fig11_quality.tsv"
VB = RES / "T7_figures/fig11_vbench.tsv"

# Table 4 covers the three I2V backbones; "Wan2.1" is the I2V model.
LABELS = [("cog", "CogVideo"), ("hunyuan", "Hunyuan"),
          ("wan_i2v", "Wan2.1")]

# (resolution, ngpu) blocks, in the order Table 4 presents them.
BLOCKS = [("2k", 1, r"2K (1 GPU)"), ("2k", 4, r"2K (4 GPUs)"), ("4k", 4, r"4K (4 GPUs)")]

# The caption (set here because the .tex is generated).
CAPTION = (
    r"Stage-2 latency and VBench (\emph{with\,/\,without} cache) and visual retention, at each "
    r"backbone's swept threshold. Workload rebalancing is disabled. Retention compares the cached "
    r"output to the no-cache output of the same pipeline and seed; both are one prompt per cell, so "
    r"VBench here is a within-row comparison rather than Table~\ref{tab:quality_benchmark}'s means."
)


def load_timings():
    """{(cfg, res, ngpu, variant): row}. Only rc=0 and only the fig11 block."""
    if not TIME.exists():
        sys.exit(f"ERROR: {TIME} not found -- run scripts/run_table4.sh")
    out = {}
    with open(TIME) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["block"] != "fig11" or r["rc"] != "0":
                continue
            out[(r["config"], r["res"], int(r["ngpu"]), r["variant"])] = r
    return out


def load_quality():
    """{(cfg, res, ngpu): {psnr, ssim, lpips}} -- retention of the cached run against its own
    no-cache reference."""
    if not QUAL.exists():
        return {}
    out = {}
    with open(QUAL) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            out[(r["config"], r["res"], int(r["ngpu"]))] = r
    return out


def load_vbench():
    """{(cfg, res, ngpu, variant): avg} for the fig11 cells.

    ONE prompt per cell -- that is what the fig11 block generated, since it exists to measure latency.
    So these are noisier than Table 3's 5- and 3-prompt means and are not comparable to them. What they
    support is the within-row comparison: cache on against cache off on the identical prompt, which is
    exactly the claim this table makes.
    """
    if not VB.exists():
        return {}
    out = {}
    with open(VB) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("avg"):
                out[(r["config"], r["res"], int(r["ngpu"]), r["variant"])] = r["avg"]
    return out


def fmt(v, spec, dash="--"):
    try:
        return format(float(v), spec)
    except (TypeError, ValueError):
        return dash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    t, q, vb = load_timings(), load_quality(), load_vbench()

    gaps = []
    body = []
    for res, ngpu, header in BLOCKS:
        tint = r"\cellcolor[HTML]{DAE8FC}" if res == "2k" else r"\cellcolor[HTML]{FFCE93}"
        body.append(f"    {tint}{header} & \\multicolumn{{6}}{{c}}{{}} \\\\")
        body.append(r"    \hline")
        for cfg, label in LABELS:
            on = t.get((cfg, res, ngpu, "cache"))
            off = t.get((cfg, res, ngpu, "nocache"))
            m = q.get((cfg, res, ngpu), {})
            if not on or not off:
                missing = [v for v, r_ in (("cache", on), ("nocache", off)) if not r_]
                gaps.append(f"{cfg} {res} {ngpu}GPU: missing {', '.join(missing)}")
                body.append(f"    {label} & -- / -- & -- & -- & -- & -- & -- \\\\")
                continue
            s_on, s_off = on.get("stage2_s"), off.get("stage2_s")
            lat = f"{fmt(s_on, '.0f')} / {fmt(s_off, '.0f')}"
            try:
                sp = f"{float(s_off) / float(s_on):.2f}$\\times$"
            except (TypeError, ValueError, ZeroDivisionError):
                sp = "--"
                gaps.append(f"{cfg} {res} {ngpu}GPU: no Stage-2 time")
            # Any row whose timing was contended is flagged: latency is the point of this table.
            if on.get("exclusive") == "0" or off.get("exclusive") == "0":
                gaps.append(f"{cfg} {res} {ngpu}GPU: measured on a NON-exclusive host")
            if not m.get("psnr"):
                gaps.append(f"{cfg} {res} {ngpu}GPU: retention not scored")
            # VBench w/ and w/o, so the reader sees the cache's quality cost on the same prompt
            # rather than one number whose baseline is elsewhere.
            v_on = vb.get((cfg, res, ngpu, "cache"))
            v_off = vb.get((cfg, res, ngpu, "nocache"))
            vcell = f"{v_on} / {v_off}" if (v_on and v_off) else "--"
            if not (v_on and v_off):
                gaps.append(f"{cfg} {res} {ngpu}GPU: VBench not scored for both variants")
            body.append(f"    {label} & {lat} & {sp} & {fmt(m.get('psnr'), '.2f')} & "
                        f"{fmt(m.get('ssim'), '.2f')} & {fmt(m.get('lpips'), '.3f')} & "
                        f"{vcell} \\\\")
        # No trailing \hline on the last block: \bottomrule draws that line itself, and the pair
        # renders as a double rule.
        if (res, ngpu) != (BLOCKS[-1][0], BLOCKS[-1][1]):
            body.append(r"    \hline")

    if gaps:
        print("Incomplete or suspect:", file=sys.stderr)
        for g in gaps:
            print(f"  {g}", file=sys.stderr)
        print(file=sys.stderr)

    tex = "\n".join([
        r"%% GENERATED by plots/table4_cache_speedup.py from",
        r"%%   results/T7_figures/{timings,fig11_quality,fig11_vbench}.tsv",
        r"%% Re-run the generator rather than editing. `--` means not measured, never zero.",
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{" + CAPTION + r"}",
        r"\label{tab:cache speedup}",
        r"\resizebox{\linewidth}{!}{",
        r"\begin{tabular}{l|cc|ccc|c}",
        r"\toprule",
        r" & \multicolumn{2}{c|}{Stage-2 Latency} & \multicolumn{3}{c|}{Visual Retention} & \\",
        r"Setting & (s, w/\,/\,w/o)$\downarrow$ & Speedup$\uparrow$ & PSNR$\uparrow$ & "
        r"SSIM$\uparrow$ & LPIPS$\downarrow$ & VBench(\%)$\uparrow$ \\ \hline",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])
    Path(a.out).write_text(tex)
    print(f"wrote {a.out}")
    print(f"{len(t)} fig11 timings, {len(q)} scored cells")


if __name__ == "__main__":
    main()
