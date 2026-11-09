# shellcheck shell=bash
# Shared helpers for every runner in scripts/. Source it; do not execute it.
#
# One measured run = one `run_one` call. It writes the exact command to <outdir>/cmd.sh (so any
# number in a TSV can be traced to the command that produced it), runs it under torchrun on the
# first N GPUs of $GPU_LIST, and judges success by the artifact rather than the exit status: a
# run counts only if it wrote a fresh .mp4.
#
# Things that are easy to get silently wrong, and how this file handles them:
#
# * Prompts contain spaces, commas and apostrophes, and the prompt text IS the input image file
#   name. Commands are serialised with printf %q and prompt lists are read with `IFS= read -r`.
# * Stage-1 latents are shared by (backbone, resolution, prompt). The w/ and w/o cache runs of a
#   comparison must start from an identical Stage 1, so the no-cache run always goes first and the
#   cached run loads its latent. FRESH_STAGE1=1 gives a run a private latent instead, so that
#   Stage 1 is generated and timed (needed for the end-to-end figure).
# * Latency is read from the pipeline's own stage timers (max over ranks), never from wall time,
#   which also includes model loading and video encoding.

set -uo pipefail
# shellcheck source=config.sh
source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

# ------------------------------------------------------------------ small utilities
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

slugify() { printf '%s' "$1" | tr -c 'A-Za-z0-9' '_' | cut -c1-60; }

# res -> "H W UPSCALE IMGDIR". 720p uses the same two-stage path with upscale factor 1, so the
# 720p row of Table 3 exercises the same code as 2K/4K.
res_dims() {
  case "$1" in
    720p) echo "720 1280 1 2k_1440x2560" ;;
    2k)   echo "1440 2560 2 2k_1440x2560" ;;
    4k)   echo "2160 3840 3 4k_2160x3840" ;;
    *)    die "unknown resolution '$1' (use 720p, 2k or 4k)" ;;
  esac
}

# The first n prompts of the frozen list for a resolution, skipping any prompt that appears in
# both lists so that no prompt is scored at two resolutions. (For the first 5 / 3 this skips
# nothing; it matters only if N is raised.)
prompts_for() {   # res n
  local res="$1" n="$2" list dup count=0 p
  case "$res" in 4k) list="$PROMPTS_4K" ;; *) list="$PROMPTS_2K" ;; esac
  [ "$n" -le 0 ] && return 0
  dup=$(comm -12 <(sort "$PROMPTS_2K") <(sort "$PROMPTS_4K"))
  while IFS= read -r p; do
    [ -z "$p" ] && continue
    if [ -n "$dup" ] && grep -Fxq -- "$p" <<<"$dup"; then continue; fi
    echo "$p"
    count=$((count + 1))
    [ "$count" -ge "$n" ] && break
  done < "$list"
}

image_for() {   # res prompt -> path of the conditioning image
  local H W U IMG
  read -r H W U IMG <<<"$(res_dims "$1")"
  echo "$DATA/inputs/$IMG/$2.jpg"
}

# First n devices of $GPU_LIST, offset by $GPU_OFFSET (used to run 1-GPU cells side by side).
gpus_for() {   # n
  local n="$1" off="${GPU_OFFSET:-0}"
  local all; IFS=',' read -r -a all <<<"$GPU_LIST"
  [ $((off + n)) -le "${#all[@]}" ] || die "need $n GPU(s) at offset $off but GPU_LIST=$GPU_LIST has ${#all[@]}"
  local IFS=','; echo "${all[*]:off:n}"
}
n_gpus_available() { local all; IFS=',' read -r -a all <<<"$GPU_LIST"; echo "${#all[@]}"; }

free_port() {
  python3 - <<'PY' 2>/dev/null || echo $((20000 + RANDOM % 20000))
import socket
s = socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()
PY
}

thresh_for() {   # config -> operating-point threshold
  case "$1" in
    cog)     echo "$CACHE_THRESH_COG" ;;
    hunyuan) echo "$CACHE_THRESH_HUNYUAN" ;;
    wan_i2v) echo "$CACHE_THRESH_WAN_I2V" ;;
    wan_t2v) echo "$CACHE_THRESH_WAN_T2V" ;;
    *)       die "unknown config $1" ;;
  esac
}

# Sets the global array CF to the cache flags for <config> in <mode>:
#   nocache            no cache at all
#   ours               HyperGen's cache at the backbone's operating point: inter-step cache,
#                      intra-tile cache and region-aware control (the deployed configuration)
#   ours:<thr>         the same at an explicit threshold
#   plain              cache without region-aware control (per-tile scale factor 0): the
#                      "+ Cache" step of Figure 12
#   teacache:<thr>     TeaCache ported onto HyperGen's tiles (CogVideoX only; Table 5)
set_cache_flags() {   # config mode
  local cfg="$1" mode="$2" th
  CF=()
  case "$mode" in
    nocache) return 0 ;;
    ours)    th="$(thresh_for "$cfg")" ;;
    ours:*)  th="${mode#ours:}" ;;
    plain)   th="$(thresh_for "$cfg")" ;;
    teacache:*)
      [ "$cfg" = cog ] || die "TeaCache is ported to the CogVideoX pipeline only"
      CF=(--cache_method teacache --enable_intra_tile_cache --teacache_rel_l1_thresh "${mode#teacache:}" --teacache_storage latent)
      return 0 ;;
    *) die "unknown cache mode $mode" ;;
  esac
  case "$cfg:$mode" in
    cog:plain)       CF=(--cache_method ours --enable_intra_tile_cache --cache_thresh "$th" --static_tile_cache_scale_factor 0) ;;
    cog:*)           CF=(--cache_method ours --enable_region_aware_cache --enable_intra_tile_cache --cache_thresh "$th" --static_tile_cache_scale_factor "$STATIC_SCALE") ;;
    hunyuan:plain)   CF=(--enable-intra-tile-cache --cache-thresh "$th" --static-tile-cache-scale-factor 0) ;;
    hunyuan:*)       CF=(--enable-region-aware-cache --enable-intra-tile-cache --cache-thresh "$th" --static-tile-cache-scale-factor "$STATIC_SCALE") ;;
    wan_*:plain)     die "Wan has no per-tile scale factor, so it has no 'plain' cache step" ;;
    wan_*:*)         CF=(--enable_cache --cache_thresh "$th") ;;
  esac
}

# ------------------------------------------------------------------ per-backbone commands
# Each sets the global array CMD (argv) and CMD_DIR (working directory) for one run. The flags are
# the ones the paper's measurements used; see results/paper/manifests/*.json for the originals.
#
# HunyuanVideo: the first three flag groups are not optional. Without --flow-reverse etc. the
# pipeline exits 0 and writes a visibly corrupt video; without --two-stage-generation it ignores
# the tiling entirely; without --shift-timesteps it never shifts and leaves a seam across the frame.
HUNYUAN_DIT_WEIGHT="${HUNYUAN_DIT_WEIGHT:-ckpts/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt}"

build_cmd() {   # config res prompt ngpu outdir s1path   (uses CF)
  local cfg="$1" res="$2" p="$3" n="$4" out="$5" s1="$6"
  local H W U IMG port shift
  read -r H W U IMG <<<"$(res_dims "$res")"
  port="$(free_port)"
  shift="$(seq -s, 0 $((URS - 1)))"
  case "$cfg" in
    cog)
      CMD_DIR="$HG"
      CMD=("$COG_ENV/bin/torchrun" --nproc_per_node="$n" --master_port "$port" CogvideoI2V/pipeline.py
        --prompt "$p" --model_path "$COG_CKPT" --generate_type i2v
        --image_or_video_path "$DATA/inputs/$IMG/$p.jpg"
        --width "$W" --height "$H" --upscale_factor "$U"
        --num_inference_steps "$STEPS" --upscale_res_steps "$URS" --guidance_scale 6.0
        --num_frames "$FRAMES" --fps 8 --seed "$SEED"
        --output_path "$out/video.mp4" --low_res_latents_path "$s1"
        --loop_step "$LOOP_STEP" --shift_timesteps "$shift"
        --detail_reinject_scale 2.0 --attn_scale_coef 1.0
        "${CF[@]}") ;;
    hunyuan)
      CMD_DIR="$HG/HunyuanVideoI2V"
      CMD=("$HUNYUAN_ENV/bin/torchrun" --nproc_per_node="$n" --master_port "$port" sample_image2video.py
        --model HYVideo-T/2 --i2v-mode --i2v-resolution 720p --flow-reverse --flow-shift 7.0
        --i2v-stability --embedded-cfg-scale 6.0
        --dit-weight "$HUNYUAN_DIT_WEIGHT" --i2v-dit-weight "$HUNYUAN_DIT_WEIGHT"
        --two-stage-generation --rope-mode local --shift-timesteps "$shift"
        --prompt "$p" --i2v-image-path "$DATA/inputs/$IMG/$p.jpg"
        --video-size "$H" "$W" --video-length "$FRAMES" --infer-steps "$STEPS"
        --upscale-factor "$U" --upscale-res-steps "$URS" --seed "$SEED"
        --save-path "$out" --loop-step "$LOOP_STEP"
        --save-intermediate --output-dir "$out" --load-prev-latents-path "$s1"
        "${CF[@]}") ;;
    wan_i2v)
      CMD_DIR="$HG"
      CMD=("$WAN_ENV/bin/torchrun" --nproc_per_node="$n" --master_port "$port" WanI2V/pipeline.py
        --prompt "$p" --image_path "$DATA/inputs/$IMG/$p.jpg" --ckpt_dir "$WAN_I2V_CKPT"
        --width "$W" --height "$H" --upscale_factor "$U"
        --num_frames "$FRAMES" --num_inference_steps "$STEPS" --upscale_res_steps "$URS" --seed "$SEED"
        --output_path "$out/out.mp4" --stage1_latents_path "$s1" --rope_mode local
        --loop_step "$LOOP_STEP"
        "${CF[@]}") ;;
    wan_t2v)
      CMD_DIR="$HG"
      CMD=("$WAN_ENV/bin/torchrun" --nproc_per_node="$n" --master_port "$port" WanI2V/pipeline_t2v.py
        --prompt "$p" --task t2v-14B --ckpt_dir "$WAN_T2V_CKPT"
        --width "$W" --height "$H" --upscale_factor "$U"
        --num_frames "$FRAMES" --num_inference_steps "$STEPS" --upscale_res_steps "$URS" --seed "$SEED"
        --output_path "$out/out.mp4" --stage1_latents_path "$s1" --loop_step "$LOOP_STEP"
        "${CF[@]}") ;;
    *) die "unknown config $cfg" ;;
  esac
}

# Write a self-contained, re-runnable script for one command.
#   emit_script <file> <workdir> <log> ENV=VAL ... -- argv...
emit_script() {
  local f="$1" wd="$2" lg="$3"; shift 3
  local envs=()
  while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  {
    echo '#!/usr/bin/env bash'
    echo "# generated by scripts/lib.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ); re-run with: bash $(basename "$f")"
    printf 'cd %q || exit 1\n' "$wd"
    printf 'exec env'
    printf ' %q' "${envs[@]}"
    printf ' \\\n  '
    printf '%q ' "$@"
    printf '\\\n  > %q 2>&1\n' "$lg"
  } > "$f"
}

# The output video of a run: the newest .mp4 in the run directory that is not the Stage-1 preview
# and is newer than the run's start (a stale file from an earlier attempt must not count).
find_video() {   # outdir start_epoch
  find "$1" -maxdepth 2 -name '*.mp4' -not -name 'stage1_*' -size +100000c -newermt "@$2" \
    -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-
}

# run_one <config> <res> <prompt> <ngpu> <outdir> [ENV=VAL ...]
# Uses the global CF (set_cache_flags). Sets RUN_RC, RUN_WALL, RUN_VIDEO, RUN_GPUS.
run_one() {
  local cfg="$1" res="$2" p="$3" n="$4" out="$5"; shift 5
  local extra_env=("$@")
  local slug s1 img
  slug="$(slugify "$p")"
  img="$(image_for "$res" "$p")"
  if [ "$cfg" != wan_t2v ] && [ ! -f "$img" ]; then die "missing input image: $img"; fi
  mkdir -p "$out" "$WORK/stage1"
  if [ "${FRESH_STAGE1:-0}" = 1 ]; then
    s1="$WORK/stage1/fresh_$(basename "$out").pt"
    rm -f "$s1"
  else
    s1="$WORK/stage1/${cfg}_${res}_${slug}.pt"
  fi
  # Every measured run must LOAD its Stage-1 sketch. A run that generates Stage 1 in-process draws
  # Stage 2's re-noising noise from a generator that Stage 1 has already advanced, while a run that
  # loads the sketch starts from a fresh generator. The two Stage-2 trajectories then differ even
  # without a cache, and a retention comparison (Table 4, Table 5) would measure that difference
  # instead of the cache's. (Measured: PSNR 28.9 dB fresh-vs-loaded, against 37.1 dB loaded-vs-loaded
  # in the paper.) So the first time a sketch is needed, a preparatory no-cache run generates and
  # saves it; its own output is discarded. The paper's runs always found the sketch already saved.
  if [ "${FRESH_STAGE1:-0}" != 1 ] && [ ! -s "$s1" ] && [ "${DRY:-0}" != 1 ]; then
    local prep="$out/_stage1_prep" pn saved_cf=("${CF[@]}")
    pn=$(( n > 4 ? n : ($(n_gpus_available) >= 4 ? 4 : n) ))
    log "      first use of this Stage-1 sketch: generating it in a preparatory run (${pn} GPU, output discarded)"
    mkdir -p "$prep"; CF=()
    RUN_GPUS="$(gpus_for "$pn")"
    build_cmd "$cfg" "$res" "$p" "$pn" "$prep" "$s1"
    emit_script "$prep/cmd.sh" "$CMD_DIR" "$prep/run_log.log" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      CUDA_VISIBLE_DEVICES="$RUN_GPUS" TP_SIZE="$pn" LOW_RES_SAVE_PATH="$s1" WAN_REPO="$WAN_REPO" -- "${CMD[@]}"
    bash "$prep/cmd.sh"
    CF=("${saved_cf[@]}")
    [ -s "$s1" ] || warn "the preparatory run did not write $s1 (see $prep/run_log.log); the measured run will generate Stage 1 itself"
    find "$prep" -name '*.mp4' -delete 2>/dev/null
  fi
  RUN_GPUS="$(gpus_for "$n")"
  build_cmd "$cfg" "$res" "$p" "$n" "$out" "$s1"
  local envs=(PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$RUN_GPUS"
              TP_SIZE="$n" LOW_RES_SAVE_PATH="$s1" WAN_REPO="$WAN_REPO" "${extra_env[@]}")
  emit_script "$out/cmd.sh" "$CMD_DIR" "$out/run_log.log" "${envs[@]}" -- "${CMD[@]}"
  if [ "${DRY:-0}" = 1 ]; then
    log "DRY  $cfg $res ${n}GPU [${CF[*]:-nocache}] ${extra_env[*]:-} :: $p"
    RUN_RC=0; RUN_WALL=0; RUN_VIDEO=""
    return 0
  fi
  local t0 started
  started=$(( $(date +%s) - 2 ))
  t0=$SECONDS
  bash "$out/cmd.sh"
  RUN_RC=$?
  RUN_WALL=$((SECONDS - t0))
  echo "exit=$RUN_RC" >> "$out/run_log.log"
  RUN_VIDEO="$(find_video "$out" "$started")"
  if [ -z "$RUN_VIDEO" ]; then
    [ "$RUN_RC" -eq 0 ] && RUN_RC=90          # exited 0 but produced nothing
    warn "no output video in $out (rc=$RUN_RC); last log lines:"
    tail -5 "$out/run_log.log" | sed 's/^/    /' >&2
  fi
  [ "${FRESH_STAGE1:-0}" = 1 ] && rm -f "$s1"
  return 0
}

# ------------------------------------------------------------------ parsing a finished run
# "stage1<TAB>stage2<TAB>upscale" in seconds, from the pipeline's own timers. Max over ranks: a
# phase ends when its slowest rank does. A Stage-1 line tagged REUSED timed a torch.load, not a
# generation, and is reported as blank.
stage_times() {   # outdir
  # Only the run's own log: a preparatory Stage-1 run keeps its log in a subdirectory and must not count.
  local d="$1"
  _pick() { grep -ohE "$1" "$d/run_log.log" 2>/dev/null | grep -v REUSED \
              | grep -oE '[0-9.]+$' | sort -g | tail -1; }
  local s1 s2 up
  s1="$(_pick 'First Stage Running time: [0-9.]+|Stage 1 time: [0-9.]+')"
  s2="$(_pick 'Second Stage Running time: [0-9.]+|Stage 2 time: [0-9.]+')"
  up="$(_pick 'Upsampling Running time: [0-9.]+|Upscale latents time: [0-9.]+')"
  printf '%s\t%s\t%s\n' "${s1:-}" "${s2:-}" "${up:-}"
}

# Fraction of tile-steps the cache skipped. CogVideoX and Wan print it; HunyuanVideo logs each
# decision, so its rate is counted.
skip_rate() {   # outdir
  local d="$1" r hit miss
  r=$(grep -ohE '(CACHE|TEACACHE|ADACACHE)_SKIP_RATE[: =]+[0-9.]+' "$d/run_log.log" 2>/dev/null \
        | tail -1 | grep -oE '[0-9.]+$')
  if [ -n "$r" ]; then printf '%s' "$r"; return; fi
  hit=$(grep -c 'Cache hit, step' "$d/run_log.log" 2>/dev/null); miss=$(grep -c 'Cache miss, step' "$d/run_log.log" 2>/dev/null)
  if [ "${hit:-0}" -gt 0 ] || [ "${miss:-0}" -gt 0 ]; then
    awk -v h="${hit:-0}" -v m="${miss:-0}" 'BEGIN{t=h+m; if (t) printf "%.4f", h/t}'
  fi
}

gpu_clocks() {   # gpu-list -> "min/max/throttled"
  nvidia-smi --query-gpu=index,clocks.sm,clocks_throttle_reasons.active --format=csv,noheader,nounits 2>/dev/null \
    | awk -F', ' -v want="$1" '
      BEGIN { n = split(want, a, ","); for (i = 1; i <= n; i++) sel[a[i]] = 1 }
      ($1 in sel) { if (min == "" || $2 < min) min = $2; if ($2 > max) max = $2
                    if ($3 != "0x0000000000000000" && $3 != "0x0000000000000001") t++ }
      END { printf "%s/%s/%s", (min == "" ? 0 : min), (max == "" ? 0 : max), t + 0 }'
}

machine_tag() {
  local gpu drv torch
  gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  torch=$("$COG_ENV/bin/python" -c 'import torch;print(torch.__version__)' 2>/dev/null)
  echo "host=$(hostname) gpu=${gpu:-none} x$(nvidia-smi -L 2>/dev/null | wc -l) driver=${drv:-none} torch(cogvideo)=${torch:-n/a} artifact=$(git -C "$AE_ROOT" rev-parse --short HEAD 2>/dev/null) date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}

# ------------------------------------------------------------------ bookkeeping
tsv_init() {   # file header...
  local f="$1"; shift
  mkdir -p "$(dirname "$f")"
  if [ ! -s "$f" ]; then (IFS=$'\t'; echo "$*") > "$f"; fi
}
tsv_row() {    # file fields...
  local f="$1"; shift
  (IFS=$'\t'; echo "$*") >> "$f"
}

# Copy a finished run's video under its canonical key, plus a JSON manifest (command, config,
# git commit, GPU, sha256) so the video can be traced and re-run without this script.
collect() {   # subdir key outdir [extra-json]
  local sub="$1" key="$2" out="$3" extra="${4:-}"
  local dst="$WORK/videos/$sub/$key.mp4"
  mkdir -p "$(dirname "$dst")" "$RESULTS/manifests"
  [ -n "${RUN_VIDEO:-}" ] && [ -s "$RUN_VIDEO" ] || return 1
  cp -f "$RUN_VIDEO" "$dst"
  python3 - "$RESULTS/manifests/$key.json" "$key" "$dst" "$out/cmd.sh" "$RUN_GPUS" \
            "$(git -C "$AE_ROOT" rev-parse HEAD 2>/dev/null)" "$extra" <<'PY'
import hashlib, json, socket, subprocess, sys, datetime
path, key, vid, cmd, gpus, commit, extra = sys.argv[1:8]
try:
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.splitlines()[0]
except Exception:
    gpu = ""
m = {"key": key, "video": vid, "video_sha256": hashlib.sha256(open(vid, "rb").read()).hexdigest(),
     "host": socket.gethostname(), "gpus": gpus, "gpu_model_and_driver": gpu,
     "artifact_commit": commit, "generated_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
     "command": open(cmd).read()}
if extra:
    m.update(json.loads(extra))
json.dump(m, open(path, "w"), indent=2)
PY
}

# A run is skipped when its collected video already exists, so an interrupted runner resumes
# where it stopped. FORCE=1 re-runs everything.
already_done() { [ "${FORCE:-0}" != 1 ] && [ -s "$WORK/videos/$1/$2.mp4" ]; }

# ------------------------------------------------------------------ CLI helpers
# Parse the flags every runner accepts. Sets MODE (quick|full), DRY, MODELS_OVERRIDE, SKIP_GEN,
# SKIP_SCORE.
parse_common_args() {
  MODE=quick; DRY="${DRY:-0}"; MODELS_OVERRIDE=""; SKIP_GEN=0; SKIP_SCORE=0
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --quick) MODE=quick ;;
      --full)  MODE=full ;;
      --dry-run) DRY=1 ;;
      --models) MODELS_OVERRIDE="$2"; shift ;;
      --score-only) SKIP_GEN=1 ;;
      --no-score) SKIP_SCORE=1 ;;
      -h|--help) sed -n '2,/^set -uo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
      *) die "unknown argument: $1 (try --help)" ;;
    esac
    shift
  done
  export DRY
}

require_gpus() {   # n
  local have; have="$(n_gpus_available)"
  [ "$have" -ge "$1" ] || die "this experiment needs $1 GPUs; GPU_LIST=$GPU_LIST has $have"
}

banner() {   # title
  echo "================================================================================"
  echo " $1   (mode: $MODE$([ "${DRY:-0}" = 1 ] && echo ', DRY RUN: commands are written, nothing is executed'))"
  echo " $(machine_tag)"
  echo " results -> $RESULTS    videos -> $WORK/videos"
  echo "================================================================================"
}

# ------------------------------------------------------------------ the latency experiments
# Table 4 and Figures 11-14 all write rows of one schema into $RESULTS/T7_figures/timings.tsv,
# distinguished by the `block` column (internal names, shared by the paper's data in results/paper/
# so that the same generators read both):
#   fig10 -> Figure 11 (end-to-end)      fig11 -> Table 4 (cache)      fig12 -> Figure 12 (breakdown)
#   fig13 -> Figure 13 (scalability)     fig14 -> Figure 14 (rebalance)
T7_HEADER=(block config res variant ngpu prompt host wall_s stage1_s upscale_s stage2_s skip_rate rc collected clocks exclusive thresh rebalance)

# measure_t7 <block> <config> <res> <variant> <ngpu> <prompt> <cache-mode> [ENV=VAL ...]
measure_t7() {
  local blk="$1" cfg="$2" res="$3" var="$4" n="$5" p="$6" mode="$7"; shift 7
  local env=("$@")
  local tsv="$RESULTS/T7_figures/timings.tsv"
  local key="${blk}__${cfg}__${res}__${var}__${n}gpu__$(slugify "$p" | cut -c1-40)"
  if already_done T7 "$key"; then log "skip  $key (already measured)"; return 0; fi
  tsv_init "$tsv" "${T7_HEADER[@]}"
  set_cache_flags "$cfg" "$mode"
  local out="$WORK/runs/T7/$key"
  log "run   $blk  $cfg $res $var ${n}GPU ${env[*]:-}"
  run_one "$cfg" "$res" "$p" "$n" "$out" "${env[@]}"
  [ "$DRY" = 1 ] && return 0
  local st s1 s2 up sk clk coll=0 reb=0 th=""
  st="$(stage_times "$out")"
  s1="$(cut -f1 <<<"$st")"; s2="$(cut -f2 <<<"$st")"; up="$(cut -f3 <<<"$st")"
  sk="$(skip_rate "$out")"; clk="$(gpu_clocks "$RUN_GPUS")"
  case " ${env[*]:-} " in *ENABLE_REDISTRIBUTE=1*) reb=1 ;; esac
  [ "$mode" != nocache ] && th="$(thresh_for "$cfg")"
  if [ "$RUN_RC" -eq 0 ] && collect T7 "$key" "$out" \
       "{\"block\": \"$blk\", \"config\": \"$cfg\", \"res\": \"$res\", \"variant\": \"$var\", \"ngpu\": $n, \"cache_mode\": \"$mode\", \"env\": \"${env[*]:-}\"}"; then
    coll=1
  fi
  tsv_row "$tsv" "$blk" "$cfg" "$res" "$var" "$n" "$p" "$(hostname)" "$RUN_WALL" "$s1" "$up" "$s2" \
    "$sk" "$RUN_RC" "$coll" "$clk" 1 "$th" "$reb"
  log "      rc=$RUN_RC  stage2=${s2:-n/a}s  skip=${sk:-n/a}  wall=${RUN_WALL}s"
}

# ------------------------------------------------------------------ scoring
# PSNR/SSIM/LPIPS of <test> against <ref>, as "psnr,ssim,lpips".
retention() {   # ref test
  CUDA_VISIBLE_DEVICES="$(GPU_OFFSET=0 gpus_for 1)" "$METRICS_ENV/bin/python" "$AE_ROOT/eval/metrics.py" \
    --ref "$1" --test "$2" --csv 2>/dev/null | grep -E '^[0-9]' | tail -1
}

# VBench (the paper's five aspects) over a set of videos, appended as one TSV row.
#   vbench_cell <out.tsv> "<k=v> <k=v> ..." <video>...
vbench_cell() {
  local out="$1" fields="$2"; shift 2
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$(GPU_OFFSET=0 gpus_for 1)" "$VBENCH_ENV/bin/python" "$AE_ROOT/eval/vbench_cells.py" \
    --out "$out" --fields $fields --work "$WORK/vbench" --videos "$@"
}

# Has <out.tsv> already got a scored row whose first columns equal the given values?
scored_already() {   # out.tsv avg-col values...
  local f="$1" col="$2"; shift 2
  [ "${FORCE:-0}" != 1 ] && [ -s "$f" ] || return 1
  awk -F'\t' -v c="$col" -v want="$(IFS=$'\t'; echo "$*")" -v n="$#" '
    NR > 1 { k = $1; for (i = 2; i <= n; i++) k = k "\t" $i; if (k == want && $c != "") f = 1 }
    END { exit !f }' "$f"
}
