# HyperGen: PPoPP 2027 Artifact

**Paper:** *HyperGen: Empowering Ultra-high-resolution Video Generation Framework with Sketch-Tile
Collaboration*, PPoPP 2027.
**Archived artifact (DOI):** [10.5281/zenodo.24079619](https://doi.org/10.5281/zenodo.24079619) (v1.0, the evaluated version).
**Badges sought:** Artifacts Available, Artifacts Evaluated (Functional / Reusable).

HyperGen generates 2K and 4K video with pretrained 720p video diffusion models and no retraining.
A 720p *sketch* is generated first (Stage 1). It is upsampled, re-noised, and refined tile by tile at
the target resolution (Stage 2). Three mechanisms make Stage 2 fast:

- a fine-grained, region-aware cache that skips tile-steps whose output barely changes;
- communication-minimized tile parallelism across GPUs;
- cache-guided workload rebalancing.

This artifact contains:

| | |
|---|---|
| `hypergen/` | the implementation, for CogVideoX-1.5 (5B), HunyuanVideo-I2V (13B), and Wan2.1 I2V/T2V (14B) ([what is where](hypergen/VENDORED.md)) |
| `scripts/` | one runner per table and figure of the evaluation (Section 8), a getting-started test, and a pre-flight check |
| `eval/` | the scoring pipeline: VBench (5 aspects) and PSNR / SSIM / LPIPS |
| `plots/` | the scripts that turn measurements into the paper's tables and figures |
| `results/paper/` | the raw measurements behind the tables and figures of the evaluation ([README](results/paper/README.md)) |
| `data/` | the frozen prompt sets and conditioning images ([README](data/prompts/README.md)) |
| `env/` | environment setup and model download |

```
run_*.sh  ──>  results/<run>/*.tsv  ──>  plots/make_all.sh  ──>  out/<run>/{tables/*.tex, figures/*.pdf, summary.md}
                (stage timers,                (same generators for the paper's data
                 VBench, PSNR/SSIM/LPIPS)      in results/paper and for your runs)
```

---

## 1. Requirements

| | Getting started and `--quick` runs | Everything in the paper (`--full`) |
|---|---|---|
| GPUs | 4x NVIDIA H100 80 GB (8 for `run_all.sh --quick`) | 8x H100 80 GB, one node, NVLink |
| Weights | CogVideoX1.5-5B-I2V, ~35 GB | + HunyuanVideo-I2V, Wan2.1 I2V/T2V 14B, ~240 GB (+65 GB for FreeSwim) |
| Disk | ~150 GB (weights, envs, videos) | ~450 GB |
| Time | ~30 min getting started, ~3 h for every experiment in `--quick` mode | ~1.5 days |

Software: Linux x86-64, an NVIDIA driver that supports CUDA 12.4, `git`, `curl`, and `ffmpeg`
(`sudo apt-get install -y ffmpeg`). Python is managed by [uv](https://docs.astral.sh/uv/), which
`env/setup_envs.sh` installs if needed. Our test machine was a GCP `a3-megagpu-8g` (8x H100 80GB
HBM3) with Ubuntu 22.04. The paper's numbers were measured on nodes of the same GPU type.

**No such machine?** Artifact evaluators can ask us on HotCRP for SSH access to an 8x H100 node
with the environments, weights and artifact pre-installed, and a one-command runner.

Every path is configurable through environment variables (`scripts/config.sh`), e.g.
`CKPT_ROOT=/mnt/ckpts ENV_ROOT=/mnt/envs WORK=/mnt/work`. Use `GPU_LIST=0,1,2,3` to restrict the GPUs.

---

## 2. Getting started (~30 minutes)

```bash
git clone https://github.com/Dylan1229/PPoPP-27-AE-HyperGen.git && cd PPoPP-27-AE-HyperGen

bash env/setup_envs.sh cogvideo metrics vbench   # three venvs + VBench   (~10 min)
bash env/download_models.sh cog                  # CogVideoX1.5-5B-I2V      (~5-10 min)
bash scripts/getting_started.sh                  # check + one measured experiment (~15-25 min)
```

`getting_started.sh` does the following:

1. Checks the GPUs, environments, and weights (`scripts/check.sh`, which also prints which
   experiments can run with what is installed).
2. Regenerates every table and figure from the paper's raw data, without a GPU, into `out/paper/`.
3. Measures the CogVideoX 2K / 4-GPU row of **Table 4**: one 2K video without and with HyperGen's
   cache, sharing one Stage-1 sketch.
4. Scores the cached video against the uncached one (PSNR / SSIM / LPIPS), and both with VBench.
   The first VBench call downloads its scoring models (~5 GB).
5. Prints the row next to the paper's.

Expected result, on an otherwise idle 8x H100 node. The paper reports Stage 2 at 61 s with the
cache vs 106 s without (**1.74x**), PSNR 37.08, SSIM 0.96, LPIPS 0.039. Absolute seconds vary with
the machine; the speedup and retention should be close. The videos are in `work/videos/T7/`.
On our validation machine the script took 15 minutes and printed:

```
| backbone | setting | this run: s (w/ / w/o) | speedup | PSNR | LPIPS | paper: speedup | PSNR | LPIPS |
|---|---|---|---|---|---|---|---|---|
| cog | 2k 4 GPU | 61 / 105 | 1.73x | 37.04 | 0.039 | 1.74x | 37.08 | 0.039 |
```

---

## 3. Running the experiments

Every runner takes `--quick` (default: one backbone, one prompt, the smallest configuration that
exercises the experiment end to end) or `--full` (the paper's configuration). It writes TSVs to
`results/ae/` and videos to `work/videos/`. After any runner:

```bash
bash plots/make_all.sh --results results/ae   # -> out/ae/tables/*.tex, out/ae/figures/*.pdf, out/ae/summary.md
```

`out/ae/summary.md` lists every measured number next to the paper's. `out/<run>/tables/tables.tex`
typesets all generated tables (`pdflatex tables.tex`).

The runners resume: a finished run is skipped, so an interrupted runner can simply be restarted
(`FORCE=1` re-runs).

**Stage-1 sketches are shared.** All runs of one backbone, resolution, and prompt start from the same
saved 720p sketch (`work/stage1/`), so a comparison such as cache on vs. off differs only in Stage 2.
The first time a sketch is needed, the runner generates it in a *preparatory* no-cache run, whose
video is discarded. The log says so. This costs one extra run per sketch (2.5–12 min).
It is required for retention measurements: a run that generates Stage 1 itself re-noises Stage 2
from a different random state than a run that loads the sketch. Comparing the two gives ~29 dB
instead of ~37 dB even without a cache. The paper's runs always loaded a saved sketch.
Figure 11 needs Stage 1's duration, so it times Stage 1 in one extra run per backbone and resolution
that generates a fresh sketch, and its three bars load the shared one. `--dry-run` writes each command to `work/runs/.../cmd.sh` without running it.
Every run directory keeps its exact command (`cmd.sh`) and log (`run_log.log`), and every video gets
a JSON manifest in `results/ae/manifests/`.

`bash scripts/run_all.sh --quick` runs everything below in quick mode (~3 h on 8 GPUs, CogVideoX
weights only). `--full` runs the paper's configurations.

| Paper | What it measures | Command | `--quick` | `--full` |
|---|---|---|---|---|
| **Table 3** | VBench quality of HyperGen at 720p, 2K and 4K, without and with the cache | `scripts/run_table3.sh` | CogVideoX 2K, 1 prompt, 15 min | 3 backbones x {720p, 2K, 4K} x {w/o, w/ cache}, 5/5/3 prompts, ~12 h |
| **Table 3** | VBench quality and latency of the CineScale and FreeSwim baselines on the same backbone | `scripts/run_table3_baselines.sh` | CineScale 2K, 1 prompt, 30 min (`--with-freeswim`: + one FreeSwim 2K video, ~3.5 h on 1 GPU) | CineScale 2K/4K, FreeSwim 2K, ~7.5 h |
| **Table 4** | Stage-2 speedup from the cache, and the retention (PSNR / SSIM / LPIPS) against the uncached video | `scripts/run_table4.sh` | CogVideoX 2K, 1 and 4 GPUs, 30 min | 3 backbones, 2K (1, 4 GPUs) and 4K (4 GPUs), ~4 h |
| **Table 5** | HyperGen's cache vs. TeaCache, each at a conservative and an aggressive threshold | `scripts/run_table5.sh` | all 5 rows, 20 min, 5 GPUs | adds the third sweep point of each method |
| **Figure 11** | end-to-end latency: 1 GPU, + tile parallelism, + cache | `scripts/run_fig11.sh` | CogVideoX 2K, 20 min | 3 backbones, 2K (4 GPUs) and 4K (8 GPUs), ~6 h |
| **Figure 12** | Stage-2 latency as each mechanism is added (CogVideoX 4K) | `scripts/run_fig12.sh` | the four 4-GPU steps, 30 min | + the 1-GPU baseline, 50 min |
| **Figure 13** | Stage-2 latency against the GPU count (tile parallelism) | `scripts/run_fig13.sh` | CogVideoX 2K on 1/2/4 GPUs, 20 min | CogVideoX + HunyuanVideo, 2K and 4K up to 8 GPUs, ~3 h |
| **Figure 14** | Stage-2 latency at 4K without and with cache-guided rebalancing | `scripts/run_fig14.sh` | CogVideoX 4K, 4 GPUs, 15 min | CogVideoX + HunyuanVideo on 2/4/8 GPUs, ~2 h |

Each `--full` run needs the corresponding weights and environments (`bash scripts/check.sh` lists
what is missing): `env/setup_envs.sh` (all five main envs) and
`env/download_models.sh cog hunyuan wan_i2v wan_t2v`. The baselines additionally need
`env/setup_envs.sh cinescale freeswim` and `env/download_models.sh wan_t2v wan_t2v_diffusers`.

### What each experiment measures, and what to look for

All runs use seed 0, 41 frames, 50 denoising steps (35 in Stage 2), loop step 16 (shift every Stage-2
step), and 720p tiles: 2K is 4 tiles, 4K is 9. The cache thresholds are per backbone (CogVideoX 0.20,
HunyuanVideo 0.06, Wan2.1 I2V 0.15, Wan2.1 T2V 0.20) and the region-aware scale factor is 0.5. All
of these are in `scripts/config.sh`. **Latency is always read from the pipeline's own Stage-2 timer**
(max over ranks), except in Figure 11, which is end-to-end.

- **Table 3.** Per cell, one VBench run over the cell's prompts: 5 prompts at 720p and 2K, 3 at 4K,
  all on 4 GPUs. 4K videos are downscaled to a 1280-pixel long side before scoring, because VBench's
  backbones cannot hold native 4K frames (Section 8, "Metrics"). Expect averages close to each
  other across resolutions and between w/o and w/ cache. Single-prompt (`--quick`) scores are noisy
  and not comparable to 5-prompt means. The latency column is the median Stage-2 time of the cell.
  For the baselines it is the high-resolution stage read from each method's progress bar.
  CineScale runs its training-free arm. FreeSwim's released code is single-GPU only, and at 4K it
  runs out of memory on an 80 GB H100 (`--freeswim-4k` reproduces this).
- **Table 4.** Each backbone renders one prompt twice from the same Stage-1 sketch, without and with
  the cache, with rebalancing off. Retention compares the cached video with the uncached one of the
  same pipeline and seed. Expect speedups near the paper's (the cache decision is deterministic
  given the seed) and PSNR > 30 dB / LPIPS < 0.08.
- **Table 5.** CogVideoX 2K on one GPU per run. The no-cache reference runs first; then HyperGen's
  cache at thresholds 0.15 / 0.20 and TeaCache (ported onto the same tiles) at 0.10 / 0.20 run side
  by side, all from the same Stage-1 sketch. Expect HyperGen-slow and TeaCache-slow at equal LPIPS
  with HyperGen faster, and TeaCache-fast fastest but with the worst LPIPS.
- **Figure 11.** Origin (1 GPU, no cache), +Parallelism (4 GPUs at 2K, 8 at 4K, rebalance on),
  +Parallelism +Cache. End-to-end = Stage 1 + upsampling + Stage 2. Stage 1 runs on one GPU in all
  settings, so it is timed once (variant `stage1`) and added to every bar. It bounds the end-to-end
  speedup (Section 8.1).
- **Figure 12.** CogVideoX 4K: 1 GPU, then 4-GPU tile parallelism, then the cache with a uniform
  threshold, then region-aware control, then rebalancing. Stage 2 only.
- **Figure 13.** Stage 2 against the GPU count, with no cache and no rebalancing, annotated with
  the speedup over 1 GPU.
- **Figure 14.** Stage 2 at 4K with the cache on, without and with cache-guided rebalancing.

**Timing.** Absolute seconds depend on GPU clocks (4-GPU runs can hit the H100 power limit), the
interconnect, and storage. Each run records its GPU clocks in the `clocks` column (min/max MHz,
throttled-GPU count). Run on an otherwise idle node; never co-schedule two runs on the same GPUs
while timing.

**Our own run.** Before submission we ran the artifact on a fresh GCP `a3-megagpu-8g` (8x H100):
- environments built from scratch;
- `getting_started.sh`;
- every runner in `--quick` mode;
- one run of each other backbone;
- one prompt each of CineScale and FreeSwim.

[`results/validation/`](results/validation/README.md) has the results, together with every command
and log.

---

## 4. Which experiments this artifact covers

**Covered (runner + scoring + generator):** Tables 3, 4 and 5, and Figures 11, 12, 13 and 14, i.e.
every experiment in the main paper's evaluation (Sections 8.1 and 8.2).
`plots/make_all.sh` also regenerates all of them from the paper's data in `results/paper/`, without
a GPU.

**Not covered:**

- **Table 6** (tile parallelism vs. xDiT on CogVideoX). Upstream xDiT (xfuser 0.4.3) supports only
  CogVideoX *text*-to-video and lacks the `ofs` embedding that CogVideoX-1.5 *image*-to-video needs,
  so the comparison cannot be re-run on the paper's checkpoint with released code. The
  tile-parallelism side of the claim is measured by Figure 13.
- **Table 1** (model limits, from the models' documentation), **Table 2** (analytical communication
  cost), and Figures 1, 2, 4, 5, 6 and 10 (illustrations).
- **Figures 3, 7, 8 and 9** (motivation profiling of noise, residual and gain similarity). They are
  characterization plots, not evaluation results.
- **The supplementary material** (threshold sweeps, ablations, user study). The data behind the
  per-backbone cache thresholds is in `results/paper/T6_threshold_sweep/`.
- **Results Reproduced** is not sought. A `--full` run produces all numbers, but takes ~1.5 days
  of an 8x H100 node.

---

## 5. Reusing HyperGen

[`docs/REUSE.md`](docs/REUSE.md) explains how to:

- run HyperGen on your own prompt or image, at any resolution that is a whole number of 720p tiles;
- change the GPU count, the cache threshold, the region-aware scale, or the tile size;
- add a backbone;
- read the output files.

The short version for one video:

```bash
source scripts/lib.sh
set_cache_flags cog ours                 # or: nocache | ours:<threshold> | plain | teacache:<threshold>
DATA=/my/data run_one cog 4k "my prompt" 4 work/runs/my_video ENABLE_REDISTRIBUTE=1
cat work/runs/my_video/cmd.sh            # the exact torchrun command, reusable on its own
```

---

## 6. Repository layout

```
hypergen/          HyperGen implementation (snapshot of SuperGen@PPoPP-dev, see VENDORED.md)
scripts/           config.sh, lib.sh, check.sh, getting_started.sh, run_table{3,4,5}.sh,
                   run_table3_baselines.sh, run_fig{11,12,13,14}.sh, run_all.sh
eval/              vbench_eval.py, vbench_cells.py (VBench), metrics.py (PSNR/SSIM/LPIPS), build_latency.py
plots/             table3_quality.py ... fig14_rebalance.py, make_all.sh, compare.py
results/paper/     the paper's measurements (TSV) + per-video manifests
data/              prompt sets and conditioning images
env/               setup_envs.sh, download_models.sh, repos.txt (pinned upstream code), requirements/
docs/REUSE.md      adapting HyperGen to new inputs, settings and models
```

Created when you run things (git-ignored): `envs/`, `ckpts/`, `third_party/`, `work/` (run
directories, videos, Stage-1 sketches), `results/ae/`, `out/`.

## License and citation

Apache-2.0 (`LICENSE`), except parts derived from third-party code, which keep their licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Model weights are downloaded from their
publishers and are not redistributed. To cite, use [`CITATION.cff`](CITATION.cff).
