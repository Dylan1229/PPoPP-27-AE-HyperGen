#!/usr/bin/env python
"""Build T5_algo/latency.tsv, the latency column of Table 3, from a results directory.

    python eval/build_latency.py --results results/ae

Table 3's latency is the tiled SECOND stage only, on 4 GPUs, excluding the shared 720p base that
every method generates first (paper, Table 3 caption). Sources:

  * HyperGen rows:  median Stage-2 time over the prompts of each (backbone, resolution, cache)
                    cell in T1_quality/timings.tsv (the quality runs, all on 4 GPUs);
  * CineScale / FreeSwim: median high-resolution-stage time in T5_algo/timings.tsv (column
                    highres_s, parsed from each baseline's own progress bar).

In the paper's own data (results/paper/T5_algo/latency.tsv) the HyperGen values were taken from the
single-prompt Table 4 runs instead; both are Stage 2 on 4 GPUs with the same configuration, so they
measure the same quantity, but a median over 5 (3) prompts is not expected to equal one prompt's time.
"""
import argparse
import csv
import os
import statistics
import sys
from collections import defaultdict


def rows(path):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    a = ap.parse_args()

    ours = defaultdict(list)
    ngpu = {}
    for r in rows(os.path.join(a.results, "T1_quality", "timings.tsv")):
        if r.get("rc") == "0" and r.get("stage2_s"):
            k = (r["config"], r["res"], r["variant"])
            ours[k].append(float(r["stage2_s"]))
            ngpu[k] = str(len(r["gpus"].split(",")))
    base = defaultdict(list)
    for r in rows(os.path.join(a.results, "T5_algo", "timings.tsv")):
        if r.get("rc") == "0" and r.get("highres_s"):
            k = (r["method"], r["res"], "nocache")
            base[k].append(float(r["highres_s"]))
            ngpu[k] = str(len(r["gpus"].split(",")))

    if not ours and not base:
        sys.exit(f"no timings under {a.results}/T1_quality or {a.results}/T5_algo")
    out = os.path.join(a.results, "T5_algo", "latency.tsv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        fh.write("config\tres\tvariant\tngpu\tstage2_s\tn_runs\tnote\n")
        for k, v in sorted(ours.items()):
            fh.write("\t".join([*k, ngpu[k], f"{statistics.median(v):.0f}", str(len(v)),
                                "median Stage-2 time of the quality runs"]) + "\n")
        for k, v in sorted(base.items()):
            fh.write("\t".join([*k, ngpu[k], f"{statistics.median(v):.0f}", str(len(v)),
                                "median high-resolution-stage time from the method's progress bar"]) + "\n")
    print(f"wrote {out} ({len(ours)} HyperGen cell(s), {len(base)} baseline cell(s))")


if __name__ == "__main__":
    main()
