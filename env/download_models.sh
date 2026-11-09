#!/usr/bin/env bash
# Download the model weights from Hugging Face into $CKPT_ROOT (default ./ckpts).
#
#   bash env/download_models.sh                 cog hunyuan wan_i2v wan_t2v   (main paper, ~240 GB)
#   bash env/download_models.sh cog             only CogVideoX (~35 GB): enough for the getting-started
#                                               test, Tables 4-5 and Figures 11-14 in --quick mode
#   bash env/download_models.sh wan_t2v wan_t2v_diffusers    the Table 3 baselines
#
# Uses the `hf` CLI from the metrics env (env/setup_envs.sh metrics). Set HF_TOKEN if your network
# requires authentication; all models are public. Re-running resumes partial downloads.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../scripts/config.sh
source "$ROOT/scripts/config.sh"

HF="${HF:-$METRICS_ENV/bin/hf}"
[ -x "$HF" ] || HF="$(command -v hf || true)"
[ -n "$HF" ] || { echo "no 'hf' CLI: run 'bash env/setup_envs.sh metrics' first" >&2; exit 1; }
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
"$(dirname "$HF")/python" -c 'import hf_transfer' 2>/dev/null || export HF_HUB_ENABLE_HF_TRANSFER=0

TARGETS=("$@")
[ "${#TARGETS[@]}" -eq 0 ] && TARGETS=(cog hunyuan wan_i2v wan_t2v)
mkdir -p "$CKPT_ROOT"
get() { echo "--> $1  ->  $2"; "$HF" download "$1" --local-dir "$2" || { echo "FAILED: $1" >&2; exit 1; }; }

for t in "${TARGETS[@]}"; do
  case "$t" in
    cog)     get THUDM/CogVideoX1.5-5B-I2V "$COG_CKPT" ;;
    hunyuan) get tencent/HunyuanVideo-I2V "$HUNYUAN_CKPT"
             # HunyuanVideo-I2V needs two text encoders next to its weights (its ckpts/README.md).
             get xtuner/llava-llama-3-8b-v1_1-transformers "$HUNYUAN_CKPT/text_encoder_i2v"
             get openai/clip-vit-large-patch14 "$HUNYUAN_CKPT/text_encoder_2"
             ln -sfn "$HUNYUAN_CKPT" "$HG/HunyuanVideoI2V/ckpts" ;;
    wan_i2v) get Wan-AI/Wan2.1-I2V-14B-720P "$WAN_I2V_CKPT" ;;
    wan_t2v) get Wan-AI/Wan2.1-T2V-14B "$WAN_T2V_CKPT" ;;
    wan_t2v_diffusers) get Wan-AI/Wan2.1-T2V-14B-Diffusers "$WAN_T2V_DIFFUSERS_CKPT" ;;
    *) echo "unknown model '$t' (cog hunyuan wan_i2v wan_t2v wan_t2v_diffusers)" >&2; exit 1 ;;
  esac
done
du -sh "$CKPT_ROOT"/* 2>/dev/null
echo "done. Next: bash scripts/check.sh"
