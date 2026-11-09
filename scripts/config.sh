# shellcheck shell=bash
# Every knob of the artifact, in one place. All values are ${VAR:-default}, so anything can be
# overridden from the environment without editing this file, e.g.
#
#   CKPT_ROOT=/mnt/ckpts ENV_ROOT=/mnt/envs bash scripts/run_table4.sh --quick
#
# Sourced by scripts/lib.sh; do not execute directly.

# ------------------------------------------------------------------ locations
AE_ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
HG="${HG:-$AE_ROOT/hypergen}"                      # the HyperGen implementation (vendored)
DATA="${DATA:-$AE_ROOT/data}"                      # frozen prompts + input images
THIRD_PARTY="${THIRD_PARTY:-$AE_ROOT/third_party}" # upstream repos cloned by env/setup_envs.sh
ENV_ROOT="${ENV_ROOT:-$AE_ROOT/envs}"              # one Python venv per backend
CKPT_ROOT="${CKPT_ROOT:-$AE_ROOT/ckpts}"           # model weights (env/download_models.sh)
RESULTS="${RESULTS:-$AE_ROOT/results/ae}"          # small, text: TSVs, logs, manifests
WORK="${WORK:-$AE_ROOT/work}"                      # large: per-run dirs, videos, Stage-1 latents
OUT_ROOT="${OUT_ROOT:-$AE_ROOT/out}"               # generated tables / figures

export WAN_REPO="${WAN_REPO:-$THIRD_PARTY/Wan2.1}"
VBENCH_REPO="${VBENCH_REPO:-$THIRD_PARTY/VBench}"
CINESCALE_REPO="${CINESCALE_REPO:-$THIRD_PARTY/CineScale}"
FREESWIM_REPO="${FREESWIM_REPO:-$THIRD_PARTY/FreeSwim}"

# ------------------------------------------------------------------ checkpoints
COG_CKPT="${COG_CKPT:-$CKPT_ROOT/CogVideoX1.5-5B-I2V}"
HUNYUAN_CKPT="${HUNYUAN_CKPT:-$CKPT_ROOT/HunyuanVideo-I2V}"
WAN_I2V_CKPT="${WAN_I2V_CKPT:-$CKPT_ROOT/Wan2.1-I2V-14B-720P}"
WAN_T2V_CKPT="${WAN_T2V_CKPT:-$CKPT_ROOT/Wan2.1-T2V-14B}"
# FreeSwim loads Wan through Diffusers, so it needs the Diffusers-format copy of the same model.
WAN_T2V_DIFFUSERS_CKPT="${WAN_T2V_DIFFUSERS_CKPT:-$CKPT_ROOT/Wan2.1-T2V-14B-Diffusers}"

# ------------------------------------------------------------------ venvs
COG_ENV="${COG_ENV:-$ENV_ROOT/cogvideo}"       # CogVideoX-1.5 (torch 2.5.1, diffusers 0.33.1)
HUNYUAN_ENV="${HUNYUAN_ENV:-$ENV_ROOT/hunyuan}" # HunyuanVideo-I2V (torch 2.4.0, flash-attn 2.6.3)
WAN_ENV="${WAN_ENV:-$ENV_ROOT/wan}"            # Wan2.1 I2V/T2V (upstream Wan2.1 requirements)
VBENCH_ENV="${VBENCH_ENV:-$ENV_ROOT/vbench}"    # VBench (pins transformers 4.33.2)
METRICS_ENV="${METRICS_ENV:-$ENV_ROOT/metrics}" # PSNR/SSIM/LPIPS + plotting
CINESCALE_ENV="${CINESCALE_ENV:-$ENV_ROOT/cinescale}"
FREESWIM_ENV="${FREESWIM_ENV:-$ENV_ROOT/freeswim}"

# ------------------------------------------------------------------ hardware
# GPUs the artifact may use, as a CUDA device list. A run on n GPUs takes the first n entries.
GPU_LIST="${GPU_LIST:-0,1,2,3,4,5,6,7}"

# ------------------------------------------------------------------ paper protocol (Section 8)
SEED="${SEED:-0}"
FRAMES="${FRAMES:-41}"             # 41 frames, 5 s at 8 fps
STEPS="${STEPS:-50}"               # total denoising steps
URS="${URS:-35}"                   # Stage-2 steps: re-noise to step 15 of 50 and run the last 35
LOOP_STEP="${LOOP_STEP:-16}"       # shift stride = 1/16 tile, shifting every Stage-2 step
STATIC_SCALE="${STATIC_SCALE:-0.5}" # region-aware scale factor alpha (Algorithm 2)

# Per-backbone cache thresholds (paper Section 8 "Workloads"; chosen by the sweep in the
# supplementary Table 8, see results/paper/T6_threshold_sweep/operating_points.env). The gate
# compares residual magnitudes, which differ by up to 3.9x between backbones, so one shared
# value does not work.
CACHE_THRESH_COG="${CACHE_THRESH_COG:-0.20}"
CACHE_THRESH_HUNYUAN="${CACHE_THRESH_HUNYUAN:-0.06}"
CACHE_THRESH_WAN_I2V="${CACHE_THRESH_WAN_I2V:-0.15}"
CACHE_THRESH_WAN_T2V="${CACHE_THRESH_WAN_T2V:-0.20}"

# Prompt sets. Table 3 averages over the first N of each frozen list (5 at 2K and 720p, 3 at 4K).
PROMPTS_2K="${PROMPTS_2K:-$DATA/prompts/eval_2k_30.txt}"
PROMPTS_4K="${PROMPTS_4K:-$DATA/prompts/eval_4k_20.txt}"
N_720P="${N_720P:-5}"
N_2K="${N_2K:-5}"
N_4K="${N_4K:-3}"

# The single prompt used by every latency experiment (Table 4, Figures 11-14), as in the paper.
FIG_PROMPT_2K="${FIG_PROMPT_2K:-a person is putting sauce on a burger}"
FIG_PROMPT_4K="${FIG_PROMPT_4K:-a bunch of houses that are on a hillside}"
# Table 4's HunyuanVideo 2K row was measured on a different prompt from the other backbones
# (results/paper/T7_figures/timings.tsv); the artifact keeps that so its rows match the paper.
FIG_PROMPT_2K_HUNYUAN="${FIG_PROMPT_2K_HUNYUAN:-a mountain range with a sky background}"
