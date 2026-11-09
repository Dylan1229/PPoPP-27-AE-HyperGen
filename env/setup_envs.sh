#!/usr/bin/env bash
# Create the Python environments and fetch the upstream code the artifact runs against.
#
#   bash env/setup_envs.sh                      cogvideo hunyuan wan vbench metrics  (main paper)
#   bash env/setup_envs.sh cogvideo metrics vbench   only what the getting-started test needs
#   bash env/setup_envs.sh cinescale freeswim   the Table 3 baselines
#
# The backends need SEPARATE environments: their torch / diffusers / transformers pins conflict
# (CogVideoX: torch 2.5.1 + diffusers 0.33.1; HunyuanVideo: torch 2.4.0 + transformers 4.39;
# VBench: transformers 4.33). Each env is a uv-managed venv under $ENV_ROOT (default ./envs);
# uv downloads the Python interpreter it needs, so no system Python 3.10/3.11 is required.
# Exact versions installed on our test machine are in env/requirements/<env>.txt.
#
# Requirements: Linux x86-64, NVIDIA driver supporting CUDA 12.4, git, curl, ffmpeg (for VBench's
# 4K downscale: `sudo apt-get install -y ffmpeg`). ~60 GB of disk for all envs.
# Idempotent: an env whose sanity import already works is skipped (FORCE=1 rebuilds).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../scripts/config.sh
source "$ROOT/scripts/config.sh"

TARGETS=("$@")
[ "${#TARGETS[@]}" -eq 0 ] && TARGETS=(cogvideo hunyuan wan vbench metrics)
mkdir -p "$ENV_ROOT" "$THIRD_PARTY"

say() { printf '\n=== %s ===\n' "$*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

# ------------------------------------------------------------------ uv
UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
if [ -z "$UV" ]; then
  say "installing uv (https://docs.astral.sh/uv/)"
  curl -LsSf https://astral.sh/uv/install.sh | sh || die "could not install uv"
  UV="$HOME/.local/bin/uv"
fi
command -v ffmpeg >/dev/null || echo "WARNING: ffmpeg not found; VBench cannot score 4K videos without it (sudo apt-get install -y ffmpeg)" >&2

TORCH_CU124="--index-url https://download.pytorch.org/whl/cu124"
mkvenv() {   # name python-version [managed]
  # "managed": a uv-managed CPython, which ships its C headers. Triton (torch.compile) compiles a
  # small C module against Python.h at first use, and a system Python without python3-dev lacks it.
  local pref=(); [ "${3:-}" = managed ] && pref=(--python-preference only-managed)
  "$UV" venv "$ENV_ROOT/$1" --python "$2" "${pref[@]}" --seed --allow-existing >/dev/null || die "venv $1"
}
pipi() {     # name pip-args...
  local n="$1"; shift
  "$UV" pip install --python "$ENV_ROOT/$n/bin/python" "$@"
}
usable() {   # name python-snippet
  [ "${FORCE:-0}" != 1 ] && "$ENV_ROOT/$1/bin/python" -c "$2" >/dev/null 2>&1
}
# Clone <url> into third_party/<name> at <ref> (a commit, tag or branch).
clone_at() {   # name url ref
  local d="$THIRD_PARTY/$1"
  if [ ! -d "$d/.git" ]; then git clone -q "$2" "$d" || die "clone $2"; fi
  git -C "$d" fetch -q --all --tags 2>/dev/null
  git -C "$d" checkout -q "$3" || die "checkout $3 in $d"
  echo "  $1 @ $(git -C "$d" rev-parse --short HEAD)"
}
# Upstream code is pinned in env/repos.txt (name|url|ref).
repo_field() {   # name column -> value (fields are trimmed; '#' lines are comments)
  awk -F'|' -v n="$1" -v c="$2" '/^[[:space:]]*#/ {next} {k = $1; gsub(/^[ \t]+|[ \t]+$/, "", k)}
    k == n {v = $c; gsub(/^[ \t]+|[ \t]+$/, "", v); print v}' "$ROOT/env/repos.txt"
}
clone_repo() { clone_at "$1" "$(repo_field "$1" 2)" "$(repo_field "$1" 3)"; }

# ------------------------------------------------------------------ envs
env_cogvideo() {
  usable cogvideo 'import torch, diffusers; assert diffusers.__version__ == "0.33.1"' && { echo "cogvideo: ok (skip)"; return; }
  mkvenv cogvideo 3.10
  pipi cogvideo -r "$HG/CogvideoI2V/requirements.txt" || die "cogvideo requirements"
}

env_hunyuan() {
  usable hunyuan 'import torch, flash_attn, diffusers' && { echo "hunyuan: ok (skip)"; return; }
  mkvenv hunyuan 3.11
  pipi hunyuan torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 $TORCH_CU124 || die "hunyuan torch"
  pipi hunyuan -r "$HG/HunyuanVideoI2V/requirements.txt" || die "hunyuan requirements"
  # Official prebuilt wheel (cp311, torch 2.4, cxx11abi FALSE); avoids compiling flash-attn.
  pipi hunyuan "https://github.com/Dao-AILab/flash-attention/releases/download/v2.6.3/flash_attn-2.6.3+cu123torch2.4cxx11abiFALSE-cp311-cp311-linux_x86_64.whl" \
    || die "hunyuan flash-attn"
}

env_wan() {
  clone_repo Wan2.1
  usable wan 'import torch, flash_attn, easydict, diffusers; assert diffusers.__version__ == "0.33.1"' && { echo "wan: ok (skip)"; return; }
  mkvenv wan 3.10
  pipi wan torch==2.5.1 torchvision==0.20.1 $TORCH_CU124 || die "wan torch"
  # Wan2.1's requirements, minus flash_attn (installed from a prebuilt wheel below).
  grep -viE '^(flash_attn|torch|torchvision)' "$WAN_REPO/requirements.txt" > "$ENV_ROOT/wan/requirements.wan.txt"
  pipi wan -r "$ENV_ROOT/wan/requirements.wan.txt" -c "$ROOT/env/constraints/wan.txt" || die "wan requirements"
  pipi wan "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    || die "wan flash-attn"
}

env_vbench() {
  clone_repo VBench
  usable vbench 'import torch, vbench' && { echo "vbench: ok (skip)"; return; }
  # VBench's setup.py rejects CUDA > 12.1 on behalf of detectron2, which only the detection
  # dimensions use; none of the paper's five aspects need it, so the check is disabled.
  sed -i 's/^\(\s*\)check_torch_version()/\1# check_torch_version()  # disabled: only detectron2 needs it/' "$VBENCH_REPO/setup.py"
  mkvenv vbench 3.10
  pipi vbench torch==2.5.1 torchvision==0.20.1 $TORCH_CU124 || die "vbench torch"
  pipi vbench -r "$VBENCH_REPO/requirements.txt" || die "vbench requirements"
  pipi vbench "setuptools<81" wheel || die "vbench setuptools"      # openai-clip imports pkg_resources
  pipi vbench --no-deps --no-build-isolation -e "$VBENCH_REPO" || die "vbench install"
}

env_metrics() {
  usable metrics 'import torch, lpips, skimage, av, matplotlib' && { echo "metrics: ok (skip)"; return; }
  mkvenv metrics 3.10
  pipi metrics torch==2.5.1 torchvision==0.20.1 $TORCH_CU124 || die "metrics torch"
  pipi metrics lpips==0.1.4 scikit-image "imageio[pyav]" imageio-ffmpeg av numpy matplotlib pandas \
    "huggingface_hub[cli,hf_transfer]" || die "metrics packages"
}

# Table 3 baselines, at the commits the paper's runs used (env/repos.txt), driven by our
# parameterised entry points in baselines/ (they pin the shared protocol: prompt, frames, seed,
# 720p base, 14B model). See baselines/*/README.md.
env_cinescale() {
  clone_repo CineScale
  # Our one change to upstream: its PyTorch attention fallback passes FlashAttention's
  # `softmax_scale=` to F.scaled_dot_product_attention, which calls it `scale=`.
  if git -C "$CINESCALE_REPO" apply --check "$ROOT/baselines/cinescale/sdpa-scale-kwarg.patch" 2>/dev/null; then
    git -C "$CINESCALE_REPO" apply "$ROOT/baselines/cinescale/sdpa-scale-kwarg.patch" && echo "  applied sdpa-scale-kwarg.patch"
  fi
  cp -f "$ROOT/baselines/cinescale/cinescale_t2v14b_ours.py" "$CINESCALE_REPO/"
  usable cinescale 'import torch, flash_attn, modelscope; from xfuser.core.long_ctx_attention import xFuserLongContextAttention' && { echo "cinescale: ok (skip)"; return; }
  mkvenv cinescale 3.10
  pipi cinescale torch==2.6.0 torchvision==0.21.0 $TORCH_CU124 || die "cinescale torch"
  grep -viE '^(torch|torchvision)([=<> ]|$)' "$CINESCALE_REPO/requirements.txt" > "$ENV_ROOT/cinescale/requirements.cs.txt"
  pipi cinescale -r "$ENV_ROOT/cinescale/requirements.cs.txt" || die "cinescale requirements"
  # xfuser 0.4.3: newer releases (0.7) import update_npu_out from yunchang, which yunchang 0.6.x lacks,
  # and CineScale's sequence-parallel code is written against the 0.4 API.
  pipi cinescale "xfuser==0.4.3" "diffusers==0.33.1" || die "cinescale extras"
  # CineScale's Wan DiT also asserts that flash-attn is importable.
  pipi cinescale "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl" \
    || die "cinescale flash-attn"
}

env_freeswim() {
  clone_repo FreeSwim
  # FreeSwim has no license file, so the artifact does not redistribute a modified copy of its code:
  # our entry point is a patch against upstream inference.py, applied here. The result is
  # byte-identical to the inference_14b_ours.py the paper's runs used.
  if [ ! -f "$FREESWIM_REPO/inference_14b_ours.py" ]; then
    patch -s -o "$FREESWIM_REPO/inference_14b_ours.py" "$FREESWIM_REPO/inference.py" \
      < "$ROOT/baselines/freeswim/freeswim-14b-ours.patch" || die "freeswim patch"
    echo "  created inference_14b_ours.py"
  fi
  usable freeswim 'import torch, diffusers, cv2, os, sysconfig; assert diffusers.__version__ == "0.35.2"
assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))' && { echo "freeswim: ok (skip)"; return; }
  # FreeSwim's driver uses torch.compile, hence a Python with headers (see mkvenv).
  [ -d "$ENV_ROOT/freeswim" ] && rm -rf "$ENV_ROOT/freeswim"
  mkvenv freeswim 3.10 managed
  # Upstream README: torch 2.7.1 and diffusers 0.35.2 (flex attention needs torch >= 2.5).
  pipi freeswim torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126 || die "freeswim torch"
  pipi freeswim diffusers==0.35.2 transformers accelerate ftfy sentencepiece opencv-python-headless \
    "imageio[ffmpeg]" imageio-ffmpeg || die "freeswim packages"
}

for t in "${TARGETS[@]}"; do
  say "$t"
  case "$t" in
    cogvideo|hunyuan|wan|vbench|metrics|cinescale|freeswim) "env_$t" ;;
    *) die "unknown env '$t' (cogvideo hunyuan wan vbench metrics cinescale freeswim)" ;;
  esac
done

# HunyuanVideo resolves its weights relative to ./ckpts inside its own directory.
if [ -d "$HUNYUAN_CKPT" ] && [ ! -e "$HG/HunyuanVideoI2V/ckpts" ]; then
  ln -sfn "$HUNYUAN_CKPT" "$HG/HunyuanVideoI2V/ckpts"
fi
say "done. Next: bash env/download_models.sh, then bash scripts/check.sh"
