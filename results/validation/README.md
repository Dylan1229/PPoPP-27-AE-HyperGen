# Our validation run of this artifact

Before submission we ran the artifact once, as a reviewer would, on a fresh cloud machine. This
directory holds everything that run produced except the videos and weights.

**Machine:** GCP `a3-megagpu-8g` (8x H100 80GB HBM3, NVLink), Ubuntu 22.04, NVIDIA driver 580.178.04.
**Date:** 2026-11-09, 00:25-07:00 UTC.
**Procedure:**

1. `env/setup_envs.sh` built all seven environments from scratch (~10 min). The exact versions
   are in [`env/requirements/`](../../env/requirements/).
2. `env/download_models.sh cog hunyuan wan_i2v wan_t2v wan_t2v_diffusers` fetched the weights.
3. `scripts/getting_started.sh` ran (15 min).
4. Every runner ran in `--quick` mode, with CogVideoX on GPUs 0-3 and the 4K experiments on GPUs 4-7.
5. Table 4 ran once more for HunyuanVideo and Wan2.1 I2V, and Table 3 ran once for Wan2.1 T2V,
   so that every backbone's code path ran at least once.
6. Table 3's baselines ran for one prompt each: CineScale (4 GPUs) and FreeSwim (1 GPU).
7. `getting_started.sh` ran a second time into a separate results directory, to check the final
   version end to end.

The exact batch scripts are in `orchestration/`.

## Results of this run

`out_ae/summary.md` has every number. `out_ae/tables/` and `out_ae/figures/` are what
`plots/make_all.sh --results results/ae` generated from this run. In brief, with Stage-2 latency
given as with / without the cache:

| Experiment | This run |
|---|---|
| Table 4, CogVideoX 2K, 1 GPU | 181 / 413 s, 2.28x, PSNR 39.62, SSIM 0.970, LPIPS 0.021 |
| Table 4, CogVideoX 2K, 4 GPUs | 61 / 105 s, 1.73x, PSNR 37.06, SSIM 0.963, LPIPS 0.039 |
| Table 4, HunyuanVideo 2K, 4 GPUs | 141 / 189 s, 1.34x, PSNR 34.25, SSIM 0.962, LPIPS 0.020 |
| Table 4, Wan2.1 I2V 2K, 4 GPUs | 409 / 489 s, 1.19x, PSNR 31.53, SSIM 0.956, LPIPS 0.047 |
| Table 5, HyperGen 0.15 / 0.20 | 1.96x / 2.29x, LPIPS 0.017 / 0.022 |
| Table 5, TeaCache 0.10 / 0.20 | 1.84x / 3.16x, LPIPS 0.017 / 0.026 |
| Figure 11, CogVideoX 2K | Origin / +Parallelism / +Parallelism +Cache: 590 / 279 / 241 s (2.45x) |
| Figure 12, CogVideoX 4K, 4 GPUs | tile parallelism / + cache / + region-aware / + rebalance: 313 / 225 / 186 / 163 s |
| Figure 13, CogVideoX 2K | 1 / 2 / 4 GPUs: 413 / 209 / 105 s (1.98x, 3.93x) |
| Figure 14, CogVideoX 4K, 4 GPUs | without / with rebalancing: 195 / 169 s (1.15x) |
| Table 3, CogVideoX 2K | VBench 87.68 without / 87.59 with the cache (1 prompt) |
| Table 3, Wan2.1 T2V 2K | 467 / 372 s; VBench 88.50 without / 87.72 with the cache (1 prompt) |
| Table 3, CineScale 2K | high-res stage 1198 s, VBench 83.90 (1 prompt) |
| Table 3, FreeSwim 2K | high-res stage 10707 s (305.9 s/it), VBench 83.83 (1 prompt) |

The second getting-started run (`results_gs/`, `out_gs/summary.md`) gave 61 / 105 s, 1.73x,
PSNR 37.04.

## What the run found, and what we fixed

These problems were found during this run. All are fixed in the submitted version:

- **Stage-1 sharing.** A run that generates its Stage-1 sketch itself re-noises Stage 2 from a
  different random state than a run that loads the saved sketch. The first getting-started run
  compared such a pair, and retention came out at 28.9 dB instead of 37 dB. The runners now
  create a missing sketch in a preparatory run, so every measured run loads it, as the paper's
  runs did. Figure 11 had the same issue (its +Cache bar took 93 s instead of 67 s). It now times
  Stage 1 in a separate run.
- **Environments.** Fresh installs pulled releases that were too new:
  - Wan2.1's open-ended requirements pulled diffusers 0.41, which needs torch 2.6. Now pinned
    in `env/constraints/wan.txt`.
  - CineScale pulled xfuser 0.7, which is incompatible with yunchang 0.6. Now pinned to 0.4.3.
  - FreeSwim's `torch.compile` needs `Python.h`, which a system Python without `python3-dev`
    lacks. Its environment now uses a uv-managed Python.
- **CineScale's timer.** CineScale's high-res time was read from the wrong progress bar.

## Files

| Path | Contents |
|---|---|
| `results_ae/` | the TSVs and per-video manifests of the run (same layout as `results/paper/`) |
| `results_gs/` | the second getting-started run |
| `out_ae/`, `out_gs/` | the generated tables, figures, and `summary.md` |
| `runs_ae/`, `runs_gs/` | each run's exact command (`cmd.sh`) and log (`run_log.log`), including the preparatory Stage-1 runs (`_stage1_prep/`) |
| `logs/` | the runners' console logs and the environment setup logs |
| `orchestration/` | the batch scripts that sequenced the runners on the two GPU halves |

The runners append rows, so a cell that was measured again after a fix has two rows in its TSV,
and the generators use the last one:

- the CogVideoX 2K / 4-GPU no-cache row of `T7_figures/timings.tsv`, first measured by the first
  getting-started run before the Stage-1 fix;
- the four `fig10` rows, first measured with the old Figure 11 protocol.

Logs of runs that were later repeated were overwritten by the repeat. The exception is
`runs_ae/T5/_failed_freeswim_no_python_headers/`, which shows the FreeSwim failure fixed above.
Progress bars in the logs are collapsed to their final state.

The runs used artifact commits `b15d728` through `e8c18a5` (each log's banner names its commit).
The later commits change only documentation, environment setup, and the fixes listed above.
