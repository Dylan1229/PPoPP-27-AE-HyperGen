#!/usr/bin/env python
"""VBench evaluation over a directory of generated videos (the paper's 5 aspects).

Reports Subject Consistency, Background Consistency, Motion Smoothness,
Aesthetic Quality, and Imaging Quality, plus their unweighted average -- matching
what the paper's quality tables report.

Uses `mode='custom_input'`, which scores whatever videos are in the directory
without VBench's filename conventions. Our filenames are the prompt text, so the
prompt is recoverable from the file itself.

    $ENV_ROOT/vbench/bin/python eval/vbench_eval.py \
        --videos results/gen/ours_2k_fast/videos --name ours_2k_fast \
        --out results/gen/ours_2k_fast/vbench

Env: $ENV_ROOT/vbench (env/setup_envs.sh vbench). NOT the metrics env -- VBench pins
transformers==4.33.2, which conflicts with the generation envs.

Note on 4K: VBench's vision backbones downscale/crop high-res frames, so these
scores understate 2K/4K fidelity. Read them as a check that quality is preserved
across resolutions, not as a measure of high-resolution detail.
"""
import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile


def _probe_size(path):
    """(width, height) of a video, or (0, 0) if ffprobe cannot tell.

    (0, 0) means "do not downscale": guessing that an unreadable file is 4K and re-encoding it
    would corrupt a perfectly good 720p video, whereas skipping the downscale at worst reproduces
    the OOM, which is loud.
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
            capture_output=True, text=True, timeout=60).stdout.strip()
        w, h = out.split("x")[:2]
        return int(w), int(h)
    except Exception:
        return 0, 0


def _downscale(src, dst, max_side):
    """Re-encode with the long side at max_side, preserving aspect, frame count and rate.

    `-2` on the other axis keeps it even (H.264 needs that) and lets ffmpeg pick the value, so the
    aspect ratio is not quantised by hand. Frames are never dropped: VBench's temporal dimensions
    (motion_smoothness, subject/background consistency) are computed frame-to-frame, so a changed
    frame count would change the score for reasons unrelated to the method.
    """
    scale = f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})':flags=lanczos"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vf", scale,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p", dst],
        check=True, timeout=900)

# The five aspects the paper reports, in the paper's order.
DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "motion_smoothness",
    "aesthetic_quality",
    "imaging_quality",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="directory of .mp4 files to score")
    ap.add_argument("--name", required=True, help="run name (used for output filenames)")
    ap.add_argument("--out", required=True, help="output directory for VBench results")
    ap.add_argument("--dimensions", nargs="*", default=DIMENSIONS,
                    help=f"subset to run (default: the paper's five: {' '.join(DIMENSIONS)})")
    ap.add_argument("--full-info", default=None,
                    help="path to VBench_full_info.json (default: auto-detect in the vbench package)")
    ap.add_argument("--csv", default=None, help="also append a one-row summary to this CSV")
    ap.add_argument("--max-side", type=int, default=1280,
                    help="downscale any video whose long side exceeds this before scoring "
                         "(0 disables). 4K asks VBench for a 62.57 GiB single allocation and OOMs "
                         "an 80 GB H100; 1280 is 720p-class, above what its backbones use "
                         "internally. Applied identically to every video at a resolution, so rows "
                         "within that resolution stay comparable.")
    ap.add_argument("--scratch-dir", default=None,
                    help="where downscaled copies go (default: system temp)")
    a = ap.parse_args()

    import torch
    from vbench import VBench

    if not os.path.isdir(a.videos):
        sys.exit(f"ERROR: not a directory: {a.videos}")
    videos = [f for f in os.listdir(a.videos) if f.lower().endswith((".mp4", ".gif"))]
    if not videos:
        sys.exit(f"ERROR: no videos in {a.videos}")

    full_info = a.full_info
    if full_info is None:
        import vbench
        full_info = os.path.join(os.path.dirname(vbench.__file__), "VBench_full_info.json")
    if not os.path.isfile(full_info):
        sys.exit(f"ERROR: VBench_full_info.json not found at {full_info}")

    os.makedirs(a.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4K does not fit. VBench loads every frame of every video as one float tensor before the
    # vision backbones run: 41 frames at 2160x3840 asks for 62.57 GiB in a single allocation and
    # dies on an 80 GB H100 with 13.63 GiB free. Every 4K cell failed this way, so the 4K rows of
    # the quality table were unobtainable, not merely unmeasured.
    #
    # So downscale first, into a scratch directory. The long side goes to --max-side (default
    # 1280, which is 720p-class and what VBench's backbones -- CLIP/DINO at 224, MUSIQ, LAION
    # aesthetic -- resize to internally anyway).
    #
    # This DOES cost fidelity, and in a direction worth stating: fine 4K detail is gone before
    # imaging_quality sees it, so 4K scores understate 4K output (the paper's user study covers
    # perceived detail). What it does NOT do is bias the comparison: every method at a given
    # resolution goes through the identical downscale, so rows within a resolution stay
    # comparable. Rows across resolutions were never comparable for this reason.
    scratch = None
    videos_path = a.videos
    if a.max_side > 0:
        need = []
        for f in sorted(videos):
            w, h = _probe_size(os.path.join(a.videos, f))
            if max(w, h) > a.max_side:
                need.append((f, w, h))
        if need:
            scratch = tempfile.mkdtemp(prefix=f"vbench_{a.name}_", dir=a.scratch_dir)
            print(f"Downscaling {len(need)} video(s) whose long side exceeds {a.max_side} px "
                  f"-> {scratch}", flush=True)
            for f, w, h in need:
                dst = os.path.join(scratch, f)
                _downscale(os.path.join(a.videos, f), dst, a.max_side)
                print(f"  {f}: {w}x{h} -> long side {a.max_side}", flush=True)
            # Link the ones that did not need it, so the directory VBench sees is complete: a
            # partially-copied directory would silently score a subset and report a mean over it.
            for f in sorted(videos):
                dst = os.path.join(scratch, f)
                if not os.path.exists(dst):
                    os.symlink(os.path.abspath(os.path.join(a.videos, f)), dst)
            videos_path = scratch

    print(f"Scoring {len(videos)} videos from {videos_path} on {device}", flush=True)

    try:
        bench = VBench(device, full_info, a.out)
        bench.evaluate(
            videos_path=videos_path,
            name=a.name,
            dimension_list=list(a.dimensions),
            mode="custom_input",
        )
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)

    results_path = os.path.join(a.out, f"{a.name}_eval_results.json")
    if not os.path.isfile(results_path):
        sys.exit(f"ERROR: expected results at {results_path}")
    raw = json.load(open(results_path))

    # VBench returns {dimension: [score, per_video_details]}.
    scores = {}
    for dim in a.dimensions:
        if dim in raw:
            value = raw[dim]
            scores[dim] = float(value[0] if isinstance(value, (list, tuple)) else value)

    if not scores:
        sys.exit(f"ERROR: no dimension scores parsed from {results_path}")

    average = sum(scores.values()) / len(scores)

    print()
    for dim in a.dimensions:
        if dim in scores:
            print(f"{dim:26s} {scores[dim]:.4f}")
    print(f"{'average':26s} {average:.4f}   (over {len(scores)} dimensions, {len(videos)} videos)")

    summary = {"name": a.name, "num_videos": len(videos), **scores, "average": average}
    with open(os.path.join(a.out, f"{a.name}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    if a.csv:
        header = ["name", "num_videos"] + list(a.dimensions) + ["average"]
        exists = os.path.isfile(a.csv)
        with open(a.csv, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=header, extrasaction="ignore")
            if not exists:
                writer.writeheader()
            writer.writerow(summary)
        print(f"appended to {a.csv}")


if __name__ == "__main__":
    main()
