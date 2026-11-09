# The paper's measurements

This directory holds the raw measurements behind the tables and figures in the paper's evaluation.
`bash plots/make_all.sh` turns them into those tables and figures, without a GPU.
`scripts/run_*.sh` writes the same files, with the same columns, under `results/ae/`, so the same
generators read both.

| Paper | File(s) | Generator |
|---|---|---|
| Table 3 (VBench quality, latency) | `T1_quality/vbench.tsv`, `T5_algo/latency.tsv` | `plots/table3_quality.py` |
| Table 4 (cache speedup, retention) | `T7_figures/timings.tsv` (block `fig11`), `T7_figures/fig11_quality.tsv`, `T7_figures/fig11_vbench.tsv` | `plots/table4_cache_speedup.py` |
| Table 5 (HyperGen cache vs. TeaCache) | `T2_baselines/timings.tsv`, `T2_baselines/quality.tsv` | `plots/table5_cache_comparison.py` |
| Figure 11 (end-to-end latency) | `T7_figures/timings.tsv` (block `fig10`), `T8_stage1/timings.tsv` | `plots/fig11_e2e.py` |
| Figure 12 (technique breakdown) | `T7_figures/timings.tsv` (block `fig12`) | `plots/fig12_breakdown.py` |
| Figure 13 (scalability) | `T7_figures/fig13_fig14.tsv` | `plots/fig13_scalability.py --paper` |
| Figure 14 (workload rebalance) | `T7_figures/fig13_fig14.tsv` | `plots/fig14_rebalance.py --paper` |

`T6_threshold_sweep/` is the cache-threshold sweep that chose each backbone's operating point (the
paper's supplementary material). `operating_points.env` records the chosen values, which are the
defaults in `scripts/config.sh`.

The directory names (`T1_quality`, ...) and block names (`fig10`, `fig11`, ...) are internal labels.
They are kept so that the paper's data and a fresh run share one layout; `scripts/lib.sh` lists
which block feeds which figure. Where a cell has several rows, the generators use the last one.

`manifests/*.json` has one record per generated video: host, GPUs, GPU model and driver, the exact
command, the code commit, and the video's sha256. The videos themselves are not included.

All runs used single nodes with 8x NVIDIA H100 80GB HBM3, CUDA 12.4 wheels, and PyTorch 2.5.1
(CogVideoX, Wan2.1) or 2.4.0 (HunyuanVideo).

Notes on individual tables and figures:

- **Table 3.** The w/-cache rows use each backbone's operating point (CogVideoX 0.20, HunyuanVideo
  0.06, Wan2.1 I2V 0.15, Wan2.1 T2V 0.20). Latency is Stage 2 on 4 GPUs. For CineScale and FreeSwim
  it is the high-resolution stage, read from each method's progress bar. FreeSwim at 4K runs out
  of memory (`T5_algo/timings.tsv`). Wan2.1 I2V rows are measured, but the paper's table does not
  show them.
- **Table 4.** HunyuanVideo's 2K cell uses the prompt "a mountain range with a sky background"
  (`FIG_PROMPT_2K_HUNYUAN` in `scripts/config.sh`).
- **Figure 11.** Stage 1 runs on one GPU in every setting, so the figure adds one Stage-1 time per
  backbone (`T8_stage1/timings.tsv`) to each bar's measured upsampling and Stage 2.
