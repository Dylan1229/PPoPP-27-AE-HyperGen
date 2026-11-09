#!/usr/bin/env bash
# Pre-flight check: GPUs, environments, weights, input data. Runs no model; ~1 minute.
#
#   bash scripts/check.sh
#
# Prints one line per item and, at the end, which experiments can run with what is installed.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

pass=0; fail=0
declare -A HAVE
ok()   { printf '  [ ok ] %s\n' "$*"; pass=$((pass + 1)); }
bad()  { printf '  [MISS] %s\n' "$*"; fail=$((fail + 1)); }
item() {  # key description test...
  local k="$1" d="$2"; shift 2
  if "$@" >/dev/null 2>&1; then ok "$d"; HAVE[$k]=1; else bad "$d"; HAVE[$k]=0; fi
}

echo "== hardware"
ng=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$ng" -gt 0 ]; then
  ok "$ng GPU(s): $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | sed 's/^ *//' | tr '\n' ' ')"
else bad "no NVIDIA GPU visible (nvidia-smi)"; fi
echo "     GPU_LIST=$GPU_LIST ($(n_gpus_available) usable by the runners)"
item ffmpeg "ffmpeg / ffprobe on PATH (VBench downscales 4K videos with them)" bash -c 'command -v ffmpeg && command -v ffprobe'
df -h "$WORK" "$CKPT_ROOT" 2>/dev/null | awk 'NR>1 {print "     disk: " $4 " free on " $6}' | sort -u

echo "== environments ($ENV_ROOT)"
item cogvideo "cogvideo  (torch, diffusers 0.33.1)" "$COG_ENV/bin/python" -c 'import torch, diffusers; assert diffusers.__version__=="0.33.1"; assert torch.cuda.is_available()'
item hunyuan  "hunyuan   (torch 2.4, flash-attn)"   "$HUNYUAN_ENV/bin/python" -c 'import torch, flash_attn, loguru; assert torch.cuda.is_available()'
item wan      "wan       (torch, flash-attn, Wan2.1 runtime at $WAN_REPO)" bash -c "cd '$WAN_REPO' && '$WAN_ENV/bin/python' -c 'import torch, flash_attn, wan'"
item vbench   "vbench    (VBench)"                   "$VBENCH_ENV/bin/python" -c 'import vbench, torch'
item metrics  "metrics   (lpips, scikit-image, PyAV, matplotlib)" "$METRICS_ENV/bin/python" -c 'import lpips, skimage, av, matplotlib'
item cinescale "cinescale (Table 3 baseline; optional)" bash -c "[ -f '$CINESCALE_REPO/cinescale_t2v14b_ours.py' ] && '$CINESCALE_ENV/bin/python' -c 'import torch, flash_attn'"
item freeswim  "freeswim  (Table 3 baseline; optional)" bash -c "[ -f '$FREESWIM_REPO/inference_14b_ours.py' ] && '$FREESWIM_ENV/bin/python' -c 'import torch, diffusers, os, sysconfig; assert os.path.exists(os.path.join(sysconfig.get_paths()[\"include\"], \"Python.h\"))'"

echo "== weights ($CKPT_ROOT)"
item w_cog     "CogVideoX1.5-5B-I2V      $COG_CKPT" test -f "$COG_CKPT/transformer/config.json"
item w_hunyuan "HunyuanVideo-I2V         $HUNYUAN_CKPT" bash -c "test -f '$HUNYUAN_CKPT/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt' && test -d '$HUNYUAN_CKPT/text_encoder_i2v' && test -d '$HUNYUAN_CKPT/text_encoder_2'"
item w_wani2v  "Wan2.1-I2V-14B-720P      $WAN_I2V_CKPT" bash -c "ls '$WAN_I2V_CKPT'/diffusion_pytorch_model*.safetensors* && ls '$WAN_I2V_CKPT'/Wan2.1_VAE.pth"
item w_want2v  "Wan2.1-T2V-14B           $WAN_T2V_CKPT" bash -c "ls '$WAN_T2V_CKPT'/diffusion_pytorch_model*.safetensors* && ls '$WAN_T2V_CKPT'/Wan2.1_VAE.pth"
item w_wand    "Wan2.1-T2V-14B-Diffusers $WAN_T2V_DIFFUSERS_CKPT (FreeSwim only; optional)" test -f "$WAN_T2V_DIFFUSERS_CKPT/model_index.json"
if [ "${HAVE[w_hunyuan]}" = 1 ] && [ ! -e "$HG/HunyuanVideoI2V/ckpts" ]; then
  ln -sfn "$HUNYUAN_CKPT" "$HG/HunyuanVideoI2V/ckpts" && echo "     linked $HG/HunyuanVideoI2V/ckpts -> $HUNYUAN_CKPT"
fi

echo "== input data"
missing=0
for res in 2k 4k; do
  while IFS= read -r p; do [ -f "$(image_for "$res" "$p")" ] || { echo "     missing image: $res '$p'"; missing=1; }; done \
    < <(prompts_for "$res" 30)
done
[ "$missing" = 0 ] && ok "prompt lists and conditioning images ($(ls "$DATA"/inputs/*/*.jpg | wc -l) images)" || bad "conditioning images"

echo
echo "== what can run"
can() {   # label needs...
  local label="$1"; shift; local miss=()
  for k in "$@"; do [ "${HAVE[$k]:-0}" = 1 ] || miss+=("$k"); done
  if [ "${#miss[@]}" -eq 0 ]; then printf '  %-48s yes\n' "$label"; else printf '  %-48s no (needs: %s)\n' "$label" "${miss[*]}"; fi
}
can "getting started / --quick Tables 4-5, Figs 11-14" cogvideo metrics vbench w_cog
can "Table 3 --quick (CogVideoX)"                     cogvideo vbench metrics w_cog
can "Table 3 --full (CogVideoX, Hunyuan, Wan T2V)"    cogvideo hunyuan wan vbench metrics w_cog w_hunyuan w_want2v
can "Table 4 / Figure 11 --full (+ Wan I2V)"          cogvideo hunyuan wan vbench metrics w_cog w_hunyuan w_wani2v
can "Figures 13-14 --full (+ Hunyuan)"                cogvideo hunyuan w_cog w_hunyuan
can "Table 3 baselines: CineScale"                    cinescale vbench w_want2v
can "Table 3 baselines: FreeSwim"                     freeswim vbench w_wand
echo
echo "$pass ok, $fail missing"
