"""Table 3: VBench quality of HyperGen at 720p, 2K and 4K, with and without the cache, plus the
training-free baselines (CineScale, FreeSwim) on the Wan2.1 T2V backbone.

Reads T1_quality/vbench.tsv (one row per scored cell) and T5_algo/latency.tsv (the latency column),
and writes tables/table3_quality.tex. An unmeasured cell prints as `--` and is listed on stderr.
There is no Wan2.1 T2V 720p row: at the base resolution T2V is the base model itself.

    RESULTS_DIR=results/paper python plots/table3_quality.py
    RESULTS_DIR=results/ae    python plots/table3_quality.py --check   # report gaps only
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
from _paths import RES, TABLES, FIGS, PREV  # noqa: E402  (RESULTS_DIR / OUT_DIR)
TABLE = TABLES / "table3_quality.tex"
DATA = RES / "T1_quality/vbench.tsv"
LAT = RES / "T5_algo/latency.tsv"

# Display order and labels. Keys are (config, res, variant) as recorded in T1_quality/vbench.tsv.
MODELS = [
    ("cog", "Cogvideo"),
    ("hunyuan", "Hunyuanvideo"),
    ("wan_i2v", "Wan2.1 I2V"),
    ("wan_t2v", "Wan2.1 T2V"),
]

# Blocks emitted behind a `%` in the .tex: Wan2.1 I2V is measured, but the paper's table does not
# show it. Removing it from this set shows the block.
HIDDEN = {m.strip() for m in (
    "wan_i2v",
) if m.strip()}
SETTINGS = [
    ("720p", "nocache", "720p"),
    ("2k", "nocache", "2K w/o Cache"),
    ("2k", "cache", "2K w/ Cache"),
    ("4k", "nocache", "4K w/o Cache"),
    ("4k", "cache", "4K w/ Cache"),
]
# Cells that do not exist rather than being unmeasured.
NOT_APPLICABLE = {("wan_t2v", "720p", "nocache")}
NEED = {"720p": 5, "2k": 5, "4k": 3}
# A --quick run scores fewer prompts per cell. MIN_PROMPTS=1 accepts such cells instead of printing
# placeholders (plots/make_all.sh sets it for any results directory other than results/paper).
if os.environ.get("MIN_PROMPTS"):
    NEED = {k: int(os.environ["MIN_PROMPTS"]) for k in NEED}

# The training-free baselines run the Wan2.1 T2V backbone, so they appear in the T2V block (one row
# each: neither method has a cache). They are scored by the same VBench pipeline as every other row.
BASELINES = [
    ("cinescale", "CineScale", "CineScale (2K)"),
    ("freeswim", "FreeSwim", "FreeSwim (2K)"),
]
BASELINE_RES = [("2k", "2K"), ("4k", "4K")]

# Cells that cannot run on an 80 GB H100, with what to print instead. FreeSwim denoises the full
# canvas without tiling and runs out of memory at 4K.
UNRUNNABLE = {("freeswim", "4k"):
              r"\multicolumn{7}{c}{\emph{out of memory} (80\,GB H100)}"}

# Every baseline row is always printed; an unmeasured cell shows `--`.
PLACEHOLDER = "--"


def load_latency():
    """{(method, res): (stage2_s, ngpu)} -- the HIGH-RESOLUTION stage only, for the latency column.

    Not wall clock, and that is the fair comparison rather than the convenient one: all three methods
    first generate one 720p video and then scale it up (ours stage 1; CineScale CS_HEIGHT=720;
    FreeSwim FREESWIM_BASE_H=720). That base is identical work, is the same for every method, and can
    be precomputed or offloaded, so including it compresses the ratios toward 1 and measures the
    backbone rather than the upscaling technique.

    Only the Wan T2V rows and the two baselines have an entry: those are the runs that share a task,
    a backbone and a prompt set, so their times are comparable. The other three backbones are a
    different model each and their latencies belong in Table~\ref{tab:cache speedup}, which reports
    Stage 2.

    FreeSwim is 1 GPU because its released codebase has no multi-GPU path; the column says so rather
    than comparing a 1-GPU time against two 4-GPU ones silently.
    """
    if not LAT.exists():
        return {}
    out = {}
    with open(LAT) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("stage2_s"):
                out[(r["config"], r["res"], r.get("variant", "nocache"))] = (
                    int(float(r["stage2_s"])), r["ngpu"])
    return out

# The caption (set here because the .tex is generated).
CAPTION = (
    r"Quality of \Mname{} on VBench. \textbf{V1}: Subject Consistency, \textbf{V2}: Background "
    r"Consistency, \textbf{V3}: Motion Smoothness, \textbf{V4}: Aesthetic Quality, \textbf{V5}: "
    r"Imaging Quality. Latency is the tiled second stage on 4 GPUs, excluding the shared 720p "
    r"base that every method generates first and can precompute; the 720p rows are that base. "
    r"FreeSwim exhausts an 80\,GB H100 at 4K, and its released codebase is single-GPU only."
)


def load(path: Path) -> dict:
    if not path.exists():
        sys.exit(f"ERROR: {path} not found -- run scripts/run_table3.sh first")
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            key = (r["config"], r["res"], r["variant"])
            # Last wins: a cell re-scored after more prompts finished supersedes the earlier mean.
            rows[key] = r
    return rows


def baseline_rows(rows, gaps):
    """The baseline rows that are complete enough to print, as (key, res, label, row).

    One place decides this, because three things depend on it: the \\multirow span, whether a
    \\cline is drawn, and the rows themselves. Deciding it inside the emit loop let the span count a
    row that was then skipped.

    A row is withheld when its mean is over fewer prompts than the protocol calls for. That is a gap
    in our measurement, not a property of the method, so it belongs on stderr rather than in the
    table -- and an unlabelled short mean would be worse still, since it is then over a different
    prompt set from every other row and reads as like-for-like.
    """
    out = []
    for b, _, _ in BASELINES:
        for res, res_label in BASELINE_RES:
            if (b, res) in UNRUNNABLE:
                out.append((b, res, res_label, None))     # None -> UNRUNNABLE text
                continue
            r = rows.get((b, res, "baseline"))
            if not r or not r.get("avg"):
                gaps.append(f"{b} {res}: not scored -- placeholder row")
                out.append((b, res, res_label, {}))       # {} -> dashes
                continue
            got = int(r.get("n_videos") or 0)
            if got < NEED[res]:
                # Still running: a partial mean is over a different prompt set from every other row,
                # so the row is a placeholder rather than a number that invites comparison.
                gaps.append(f"{b} {res}: only {got}/{NEED[res]} prompts -- placeholder row")
                out.append((b, res, res_label, {}))
                continue
            out.append((b, res, res_label, r))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--out", default=str(TABLE))
    ap.add_argument("--check", action="store_true", help="report gaps without writing the table")
    a = ap.parse_args()

    rows = load(Path(a.data))
    lat = load_latency()

    gaps = []
    body = []
    for cfg, label in MODELS:
        hide = cfg in HIDDEN
        n_settings = sum(1 for r, v, _ in SETTINGS if (cfg, r, v) not in NOT_APPLICABLE)
        # The T2V block carries the baselines too, so its \multirow has to span them. Counted from
        # the same function that emits them, so a withheld row cannot inflate the span.
        if cfg == "wan_t2v":
            n_settings += len(baseline_rows(rows, []))
        pre = "    % " if hide else "    "
        body.append(f"{pre}\\multirow{{{n_settings}}}{{*}}{{{label}}}")
        first = True
        for res, var, setting_label in SETTINGS:
            if (cfg, res, var) in NOT_APPLICABLE:
                continue
            r = rows.get((cfg, res, var))
            if r and r.get("avg"):
                vals = [r[k] for k in ("v1_subject", "v2_background", "v3_motion",
                                       "v4_aesthetic", "v5_imaging")]
                cells = " & ".join(v if v else "--" for v in vals)
                avg = r["avg"]
                got = int(r.get("n_videos") or 0)
                if got < NEED[res]:
                    # Scored, but over fewer prompts than the protocol calls for. That is not the
                    # same number as a complete cell, so it is flagged rather than accepted.
                    gaps.append(f"{cfg} {res} {var}: scored over {got}/{NEED[res]} prompts")
            else:
                cells = " & ".join(["--"] * 5)
                avg = "--"
                gaps.append(f"{cfg} {res} {var}: not scored")
            lead = "      % " if hide else "      "
            first = False
            # Every row carries its Stage-2 latency, keyed on (backbone, resolution, variant). Cache
            # cells use each backbone's own operating point.
            lt = lat.get((cfg, res, var))
            lcell = f"{lt[0]}" if lt else PLACEHOLDER
            body.append(f"{lead}& {setting_label:<14} & {cells} & "
                        f"\\cellcolor{{gray!20}}\\textbf{{{avg}}} & {lcell} \\\\")

        # Baselines, appended inside the T2V block. A rule separates them from our own rows: they
        # are a different method on the same backbone, not another setting of ours.
        if cfg == "wan_t2v":
            bl = baseline_rows(rows, gaps)
            if bl:
                body.append(r"    \cline{2-9}")
            for b, res, res_label, r in bl:
                name = dict((k, lab) for k, lab, _ in BASELINES)[b]
                if r is None:
                    # Cannot run at this resolution: one spanning cell, not six dashes, because the
                    # reason is categorical rather than "not measured yet".
                    body.append(f"      & {name} {res_label:<6} & {UNRUNNABLE[(b, res)]} \\\\")
                    continue
                if not r:
                    cells = " & ".join([PLACEHOLDER] * 5)
                    avg = PLACEHOLDER
                else:
                    vals = [r[k] for k in ("v1_subject", "v2_background", "v3_motion",
                                           "v4_aesthetic", "v5_imaging")]
                    cells = " & ".join(v if v else PLACEHOLDER for v in vals)
                    avg = r["avg"]
                lt = lat.get((b, res, "nocache"))
                if lt and lt[1] != "4":
                    # Annotate any arm not measured on 4 GPUs. FreeSwim's released codebase is
                    # single-GPU only, so its time is not a 4-GPU time and must not read as one.
                    lcell = f"{lt[0]} ({lt[1]}\\,GPU)"
                elif lt:
                    lcell = f"{lt[0]}"
                else:
                    lcell = PLACEHOLDER
                body.append(f"      & {name} {res_label:<6} & {cells} & "
                            f"\\cellcolor{{gray!20}}\\textbf{{{avg}}} & {lcell} \\\\")
        # A hidden block's own \midrule is commented too, or the table gains a stray rule.
        rule = "\\midrule" if (cfg, label) != MODELS[-1] else "\\bottomrule"
        body.append(("    % " if hide else "    ") + rule)

    if gaps:
        print("Incomplete cells:", file=sys.stderr)
        for g in gaps:
            print(f"  {g}", file=sys.stderr)
        print(file=sys.stderr)

    if a.check:
        done = sum(1 for cfg, _ in MODELS for res, var, _ in SETTINGS
                   if (cfg, res, var) not in NOT_APPLICABLE
                   and rows.get((cfg, res, var), {}).get("avg"))
        total = sum(1 for cfg, _ in MODELS for res, var, _ in SETTINGS
                    if (cfg, res, var) not in NOT_APPLICABLE)
        print(f"{done}/{total} cells scored")
        return

    tex = "\n".join([
        r"%% GENERATED by plots/table3_quality.py from",
        r"%%   results/T1_quality/vbench.tsv",
        r"%% Edit the measurements, not this file: re-run the generator instead.",
        r"%% `--` means the cell is not measured yet, never a value of zero.",
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{" + CAPTION + r"}",
        r"  \label{tab:quality_benchmark}",
        r"  \resizebox{\linewidth}{!}{",
        r"\begin{tabular}{c|c|cccccc|c}",
        r"    \toprule",
        r"    Model & Setting & V1(\%) & V2(\%) & V3(\%) & V4(\%) & V5(\%) & Avg. & Latency (s)$\downarrow$ \\",
        r"    \midrule",
        *body,
        r"\end{tabular}",
        r"  }",
        r"  \vspace{-10pt}",
        r"\end{table}",
        "",
    ])
    Path(a.out).write_text(tex)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
