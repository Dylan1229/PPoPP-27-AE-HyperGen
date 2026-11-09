#!/usr/bin/env python
"""Score one *cell* (a set of videos) on the paper's five VBench aspects and append one TSV row.

    $ENV_ROOT/vbench/bin/python eval/vbench_cells.py \
        --out results/ae/T1_quality/vbench.tsv \
        --fields config=cog res=2k variant=cache \
        --videos work/videos/T1/cog__2k__cache__*.mp4

The row is `<field values> n_videos v1_subject v2_background v3_motion v4_aesthetic v5_imaging avg`,
in percent, which is the schema every table generator in plots/ reads. A cell scored twice is
appended twice; the generators take the last row ("last wins"), so a re-score supersedes.

Each aspect is VBench's own aggregate over the cell's videos (mode='custom_input'), i.e. a mean
over videos, exactly as Table 3 reports. Videos whose long side exceeds 1280 px (4K) are
downscaled first by eval/vbench_eval.py: VBench's backbones cannot hold native 4K frames (one
41-frame 4K video asks for a 62.6 GiB allocation). The paper states this in Section 8, "Metrics".
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

ORDER = ["subject_consistency", "background_consistency", "motion_smoothness",
         "aesthetic_quality", "imaging_quality"]
COLS = ["v1_subject", "v2_background", "v3_motion", "v4_aesthetic", "v5_imaging"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="TSV to append the row to")
    ap.add_argument("--fields", nargs="+", required=True, help="leading columns, as key=value")
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--work", default=None, help="scratch directory (default: system temp)")
    a = ap.parse_args()

    fields = [f.split("=", 1) for f in a.fields]
    if any(len(f) != 2 for f in fields):
        sys.exit("--fields must be key=value pairs")
    videos = [v for v in a.videos if os.path.isfile(v) and not v.endswith("__stage1.mp4")]
    if not videos:
        sys.exit("no videos to score")

    name = "__".join(v for _, v in fields)
    os.makedirs(a.work, exist_ok=True) if a.work else None
    d = tempfile.mkdtemp(prefix=f"{name}_", dir=a.work)
    # Link each video under its prompt slug (the part after the last "__"). The five aspects do not
    # read the prompt, but custom_input mode takes the file name as the prompt, so keep it readable.
    for v in videos:
        slug = os.path.basename(v)[:-4].rsplit("__", 1)[-1]
        dst = os.path.join(d, slug + ".mp4")
        i = 1
        while os.path.exists(dst):
            dst = os.path.join(d, f"{slug}_{i}.mp4"); i += 1
        os.symlink(os.path.abspath(v), dst)

    here = os.path.dirname(os.path.abspath(__file__))
    scores = os.path.join(d, "scores")
    log = os.path.join(d + "_vbench.log")
    # Downscaled 4K copies go to a sibling directory, never inside the scored directory.
    scratch = d + "_scratch"
    os.makedirs(scratch, exist_ok=True)
    print(f"[vbench] {name}: {len(videos)} video(s)", flush=True)
    with open(log, "w") as fh:
        rc = subprocess.run([sys.executable, os.path.join(here, "vbench_eval.py"),
                             "--videos", d, "--name", name, "--out", scores,
                             "--scratch-dir", scratch], stdout=fh, stderr=subprocess.STDOUT).returncode
    res = os.path.join(scores, f"{name}_eval_results.json")
    if rc != 0 or not os.path.isfile(res):
        print(open(log).read()[-3000:], file=sys.stderr)
        sys.exit(f"VBench failed for {name}; full log: {log}")

    raw = json.load(open(res))
    vals = []
    for k in ORDER:
        v = raw[k][0] if isinstance(raw[k], (list, tuple)) else raw[k]
        v = float(v)
        vals.append(v * 100 if v <= 1.0 else v)   # percent, as in Table 3
    avg = sum(vals) / len(vals)

    header = [k for k, _ in fields] + ["n_videos"] + COLS + ["avg"]
    row = [v for _, v in fields] + [str(len(videos))] + [f"{x:.2f}" for x in vals] + [f"{avg:.2f}"]
    new = not os.path.isfile(a.out) or os.path.getsize(a.out) == 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "a") as fh:
        if new:
            fh.write("\t".join(header) + "\n")
        fh.write("\t".join(row) + "\n")
    print("    " + "  ".join(f"{c}={x}" for c, x in zip(COLS + ["avg"], row[-6:])), flush=True)


if __name__ == "__main__":
    main()
