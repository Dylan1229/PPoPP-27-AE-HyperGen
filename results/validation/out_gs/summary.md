# Key numbers: `gs` vs. the paper

Latencies are Stage 2 unless stated; absolute seconds depend on the machine, so compare
speedups and retention. Single-prompt VBench is noisy.

## Table 4 - cache: Stage-2 latency w/ / w/o, speedup, retention

| backbone | setting | this run: s (w/ / w/o) | speedup | PSNR | LPIPS | paper: speedup | PSNR | LPIPS |
|---|---|---|---|---|---|---|---|---|
| cog | 2k 4 GPU | 61 / 105 | 1.73x | 37.04 | 0.039 | 1.74x | 37.08 | 0.039 |
