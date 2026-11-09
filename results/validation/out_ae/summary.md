# Key numbers: `ae` vs. the paper

Latencies are Stage 2 unless stated; absolute seconds depend on the machine, so compare
speedups and retention. Single-prompt VBench is noisy.

## Table 3 - VBench average (%)

| backbone | setting | prompts | this run | paper |
|---|---|---|---|---|
| cinescale | 2k baseline | 1 | 83.90 | 80.71 |
| cog | 2k cache | 1 | 87.59 | 84.95 |
| cog | 2k nocache | 1 | 87.68 | 84.92 |
| freeswim | 2k baseline | 1 | 83.83 | 82.93 |
| wan_t2v | 2k cache | 1 | 87.72 | 87.27 |
| wan_t2v | 2k nocache | 1 | 88.50 | 87.88 |

## Table 4 - cache: Stage-2 latency w/ / w/o, speedup, retention

| backbone | setting | this run: s (w/ / w/o) | speedup | PSNR | LPIPS | paper: speedup | PSNR | LPIPS |
|---|---|---|---|---|---|---|---|---|
| cog | 2k 1 GPU | 181 / 413 | 2.28x | 39.62 | 0.021 | 2.29x | 39.65 | 0.022 |
| cog | 2k 4 GPU | 61 / 105 | 1.73x | 37.06 | 0.039 | 1.74x | 37.08 | 0.039 |
| hunyuan | 2k 4 GPU | 141 / 189 | 1.34x | 34.25 | 0.020 | 1.34x | 34.29 | 0.020 |
| wan_i2v | 2k 4 GPU | 409 / 489 | 1.19x | 31.53 | 0.047 | 1.23x | 31.46 | 0.047 |

## Table 5 - HyperGen cache vs TeaCache (CogVideoX 2K, 1 GPU)

| method | threshold | this run: speedup | LPIPS | skip | paper: speedup | LPIPS | skip |
|---|---|---|---|---|---|---|---|
| ours | 0.15 | 1.96x | 0.017 | 0.49 | 1.99x | 0.017 | 0.49 |
| ours | 0.20 | 2.29x | 0.022 | 0.56 | 2.29x | 0.022 | 0.56 |
| teacache | 0.10 | 1.84x | 0.017 | 0.46 | 1.86x | 0.017 | 0.46 |
| teacache | 0.20 | 3.16x | 0.026 | 0.69 | 3.16x | 0.026 | 0.69 |

## Figure 11 - end-to-end latency (Stage 1 + upsampling + Stage 2)

| backbone | res | this run: origin / +par / +par+cache (s) | speedup | paper speedup |
|---|---|---|---|---|
| cog | 2k | 590 / 279 / 241 | 2.45x | 2.46x |

## Figure 12 - CogVideoX 4K, Stage-2 latency as mechanisms are added

| step | GPUs | this run (s) | skip | paper (s) |
|---|---|---|---|---|
| tilepar | 4 | 313 | - | 320 |
| cache | 4 | 225 | 0.3429 | 228 |
| regionaware | 4 | 186 | 0.4429 | 189 |
| rebalance | 4 | 163 | 0.4429 | 173 |

## Figure 13 - tile-parallel scalability (no cache), speedup over 1 GPU

| backbone | res | GPUs | this run (s) | speedup |
|---|---|---|---|---|
| cog | 2k | 1 | 413 | 1.00x |
| cog | 2k | 2 | 209 | 1.98x |
| cog | 2k | 4 | 105 | 3.93x |

## Figure 14 - workload rebalance (4K, cache on)

| backbone | GPUs | this run: w/o / w/ (s) | speedup |
|---|---|---|---|
| cog | 4 | 195 / 169 | 1.15x |
