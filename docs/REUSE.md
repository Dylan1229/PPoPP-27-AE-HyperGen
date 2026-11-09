# Reusing and adapting HyperGen

This guide covers what to change for your own inputs, settings, hardware, or model. Everything
below goes through `scripts/lib.sh`, the same code the paper's experiments use.

## 1. One video, from the command line

```bash
source scripts/lib.sh                    # loads scripts/config.sh; every variable can be overridden
set_cache_flags cog ours                 # cache mode, see below
run_one cog 2k "a mountain range with a sky background" 4 work/runs/demo ENABLE_REDISTRIBUTE=1
echo "rc=$RUN_RC video=$RUN_VIDEO"
stage_times work/runs/demo               # stage1 <TAB> stage2 <TAB> upsampling, in seconds
skip_rate work/runs/demo                 # fraction of tile-steps the cache skipped
```

`run_one <config> <res> <prompt> <n_gpus> <outdir> [ENV=VALUE ...]` writes the exact command to
`<outdir>/cmd.sh` and the log to `<outdir>/run_log.log`. `cmd.sh` runs on its own, without the
artifact's scripts, so you can copy it and edit flags directly. `DRY=1` writes it without running.

| Argument | Values |
|---|---|
| config | `cog` (CogVideoX-1.5 5B I2V), `hunyuan` (HunyuanVideo-I2V 13B), `wan_i2v`, `wan_t2v` (Wan2.1 14B) |
| res | `720p`, `2k` (2560x1440, 4 tiles), `4k` (3840x2160, 9 tiles) |
| n_gpus | 1..8; tiles are distributed over the first n devices of `$GPU_LIST` |
| cache mode (`set_cache_flags <config> <mode>`) | `nocache`; `ours` (the backbone's operating point); `ours:<threshold>`; `plain` (one uniform threshold, no region-aware control); `teacache:<threshold>` (CogVideoX only) |
| environment | `ENABLE_REDISTRIBUTE=1` turns on cache-guided workload rebalancing; `TILE_SIZE=HxW` sets the latent tile size (default `90x160`, one 720p tile) |

## 2. Your own prompts and images

Image-to-video backbones take the conditioning image `<DATA>/inputs/<2k_1440x2560|4k_2160x3840>/<prompt>.jpg`,
so the prompt is the image's file name. To use your own:

```bash
mkdir -p my/inputs/4k_2160x3840 my/prompts
cp photo.jpg "my/inputs/4k_2160x3840/a red fox in the snow.jpg"
echo "a red fox in the snow" > my/prompts/eval_4k.txt
DATA=$PWD/my PROMPTS_4K=$PWD/my/prompts/eval_4k.txt bash scripts/run_table3.sh --full --models cog
```

Each runner reads its prompts from `PROMPTS_2K` / `PROMPTS_4K` and the counts from `N_720P`, `N_2K`
and `N_4K`. The latency experiments use `FIG_PROMPT_2K` / `FIG_PROMPT_4K`. Images are resized by
the pipeline, but should already have the target aspect ratio (16:9).

## 3. Other resolutions and tile layouts

The target resolution must be a whole number of tiles: 720p tiles give 2K = 2x2 and 4K = 3x3. To add
a resolution, extend `res_dims` in `scripts/lib.sh` with `H W UPSCALE_FACTOR IMAGE_DIR`, e.g.
`5k) echo "2880 5120 4 5k_2880x5120" ;;` for a 4x4 tile grid. Stage 1 stays at 720p. (The paper and
this artifact exercise 720p, 2K and 4K only; other grids follow the same code path but are untested.) A different
tile size (more, smaller tiles; supplementary "Number of tiles") is set with `TILE_SIZE=HxW` in latent
units (VAE factor 8), and must divide the latent canvas.

## 4. Cache threshold and region-aware control

The threshold `tau` (Section 5.1) compares residual magnitudes, which differ by up to 3.9x between
backbones, so it is set per backbone: `CACHE_THRESH_COG`, `CACHE_THRESH_HUNYUAN`,
`CACHE_THRESH_WAN_I2V` and `CACHE_THRESH_WAN_T2V` in `scripts/config.sh`. The region-aware scale
factor alpha (Algorithm 2) is `STATIC_SCALE`.

To pick a threshold for a new backbone, sweep it and keep the fastest value whose LPIPS against the
no-cache output stays under 0.05. That is how the paper's operating points were chosen
(`results/paper/T6_threshold_sweep/operating_points.env`):

```bash
source scripts/lib.sh
for th in 0.05 0.10 0.15 0.20 0.25; do
  set_cache_flags cog "ours:$th"; run_one cog 2k "$FIG_PROMPT_2K" 4 "work/runs/sweep_$th"
done
set_cache_flags cog nocache; run_one cog 2k "$FIG_PROMPT_2K" 4 work/runs/sweep_ref
for th in 0.05 0.10 0.15 0.20 0.25; do
  echo "$th $(stage_times work/runs/sweep_$th | cut -f2)s $(retention work/runs/sweep_ref/video.mp4 work/runs/sweep_$th/video.mp4)"
done
```

## 5. GPU count and hardware

`GPU_LIST` selects the devices, e.g. `GPU_LIST=4,5,6,7`. Tile parallelism places one or more tiles
per GPU and gathers predicted noise with one NCCL all-gather per step (`hypergen/utils/distributed.py`).
With more GPUs than tiles the extra GPUs idle; e.g. 2K has 4 tiles. Each tile runs at the backbone's
native 720p, so peak memory is set by the tile rather than the canvas: 4K fits wherever 720p fits.
On GPUs with less than 80 GB, the 13-14B backbones may need CPU offload, which these runners do not
enable.

## 6. Adding a backbone

The three mechanisms are model-agnostic and live outside the model code:

| Mechanism | Where | What a new backbone must provide |
|---|---|---|
| Tiling + shifting | `hypergen/utils/tile_utils.py` (`TiledLatentTensor2D`, `SlidingWindowConfig`) | a denoiser call that accepts one latent tile |
| Inter-step cache + gain approximation | `check_skippable` in `hypergen/CogvideoI2V/modules/cogvideo_transformer_3d.py` (reference) | the transformer's input and output for the tile, to accumulate the latent path length and the gain (Eq. 3-5) |
| Region-aware control | `hypergen/utils/tile_std_tracker.py` | the per-tile predicted-noise standard deviation |
| Tile parallelism + rebalance | `hypergen/utils/distributed.py` | nothing; it operates on tiles |

The Wan2.1 port (`hypergen/WanI2V/tiled_stage2.py`, `tiled_i2v.py`) is the most compact example: it
wraps an unmodified upstream model (`third_party/Wan2.1`) with the tiled Stage 2. To run it from the
runners, add a `build_cmd` case to `scripts/lib.sh`, a `thresh_for` entry and a `set_cache_flags`
mapping for its CLI.

## 7. Output files

Every runner appends to tab-separated files under `$RESULTS` (default `results/ae`), with the same
columns as the paper's data in `results/paper/`:

| File | One row per | Key columns |
|---|---|---|
| `T7_figures/timings.tsv` | latency run (Table 4, Figs 11-14) | `block` (fig10=Fig 11, fig11=Table 4, fig12-14=Figs 12-14), `config`, `res`, `variant`, `ngpu`, `stage1_s`, `upscale_s`, `stage2_s`, `skip_rate`, `rc`, `clocks` |
| `T7_figures/fig11_quality.tsv` | Table 4 cell | `psnr`, `ssim`, `lpips` of the cached video vs the uncached one |
| `T1_quality/timings.tsv`, `vbench.tsv` | Table 3 video / cell | stage timers; `v1_subject` ... `v5_imaging`, `avg` (percent) |
| `T2_baselines/timings.tsv`, `quality.tsv` | Table 5 run | `method`, `thresh`, `stage2_s`, `skip_rate`, `speedup`, `psnr`, `ssim`, `lpips` |
| `T5_algo/timings.tsv`, `latency.tsv` | baseline video / Table 3 latency cell | `highres_s`; `stage2_s` |
| `manifests/<key>.json` | video | command, GPUs, GPU model, commit, sha256 |

`rc` is 0 only if the run wrote a fresh video. A run that exits 0 without output is recorded as 90.
Generators take the last row of a cell, so re-running a cell supersedes it.
