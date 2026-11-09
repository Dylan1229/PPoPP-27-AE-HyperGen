#!/usr/bin/env bash
# Table 3 (baseline rows): the training-free high-resolution video baselines CineScale and FreeSwim,
# on the SAME backbone (Wan2.1 T2V 14B), prompts, seed, frame count and resolutions as HyperGen's
# Wan2.1 T2V rows, scored by the same VBench scorer.
#
#   bash scripts/run_table3_baselines.sh --quick   CineScale 2K, first prompt (~30 min, 4 GPUs)
#   bash scripts/run_table3_baselines.sh --full    CineScale 2K x5 + 4K x3 (4 GPUs, ~10 h) and
#                                                  FreeSwim 2K x5 (1 GPU, ~3.5 h per video)
# Options: --with-freeswim      add one FreeSwim 2K video to --quick (~3.5 h on 1 GPU)
#          --freeswim-4k        also attempt FreeSwim at 4K, which the paper reports as out of
#                               memory on an 80 GB H100 (fails after ~30 min)
#          --dry-run  --score-only  --no-score
#
# Both baselines first generate a 720p video and then scale it up, like HyperGen, so the latency
# column compares only the high-resolution stage. CineScale runs its TRAINING-FREE arm (CS_LORA=0).
# FreeSwim's released code is single-GPU only and runs with CPU offload.
#
# Requires: env/setup_envs.sh cinescale freeswim   and   env/download_models.sh wan_t2v wan_t2v_diffusers
# Output:   $RESULTS/T5_algo/timings.tsv   $RESULTS/T1_quality/vbench.tsv (rows config=cinescale|freeswim)
set -uo pipefail
source "$(dirname "$0")/lib.sh"

WITH_FS=0; FS_4K=0; args=()
for x in "$@"; do
  case "$x" in
    --with-freeswim) WITH_FS=1 ;;
    --freeswim-4k)   FS_4K=1 ;;
    *) args+=("$x") ;;
  esac
done
parse_common_args "${args[@]}"
banner "Table 3: CineScale / FreeSwim baselines"

# method|res|n_prompts
case "$MODE" in
  quick) PLAN=("cinescale|2k|1"); [ "$WITH_FS" = 1 ] && PLAN+=("freeswim|2k|1") ;;
  full)  PLAN=("cinescale|2k|$N_2K" "cinescale|4k|$N_4K" "freeswim|2k|$N_2K") ;;
esac
[ "$FS_4K" = 1 ] && PLAN+=("freeswim|4k|1")

T="$RESULTS/T5_algo/timings.tsv"
tsv_init "$T" method res prompt host gpus wall_s rc collected clocks highres_s note

# The high-resolution stage's duration, in seconds, from the method's own tqdm progress bar: the
# last COMPLETED bar with $URS (35) steps. Both baselines re-noise their upscaled 720p video and run
# 35 high-resolution steps, like HyperGen (CineScale restarts at step 15 of 50, FreeSwim uses
# strength 0.7), so this skips the 50-step base-video bar and any VAE-tiling bars.
highres_seconds() {   # logfile
  python3 - "$1" "$URS" <<'PY'
import re, sys
txt = open(sys.argv[1], errors="ignore").read().replace("\r", "\n")
want = sys.argv[2]
def secs(t):
    s = 0
    for part in t.split(":"):
        s = s * 60 + int(part)
    return s
# The high-res denoising loop is the last completed bar with `want` (= URS) steps (FreeSwim); a
# method with its own step count (CineScale: 50) has no such bar, so take its longest completed bar
# (the high-res loop; the VAE-decode and video-saving bars take seconds).
best, longest = None, None
for m in re.finditer(r"(\d+)/(\d+) \[((?:\d+:)?\d+:\d+)<", txt):
    if m.group(1) == m.group(2):
        t = secs(m.group(3))
        if m.group(2) == want:
            best = t
        longest = t if longest is None else max(longest, t)
best = best if best is not None else longest
if best is not None:
    print(best)
PY
}

measure_baseline() {   # method res prompt
  local m="$1" res="$2" p="$3" H W U IMG
  read -r H W U IMG <<<"$(res_dims "$res")"
  local key="${m}__${res}__baseline__$(slugify "$p")"
  if already_done T1 "$key"; then log "skip  $key (already measured)"; return 0; fi
  local out="$WORK/runs/T5/$key" n gpus
  mkdir -p "$out"
  case "$m" in
    cinescale)
      n=4; gpus="$(gpus_for 4)"
      [ -f "$CINESCALE_REPO/cinescale_t2v14b_ours.py" ] || die "CineScale driver not found in $CINESCALE_REPO (run env/setup_envs.sh cinescale)"
      # The driver loads Wan2.1-T2V-14B from models/ inside its checkout; link our download there.
      mkdir -p "$CINESCALE_REPO/models/Wan-AI"
      [ -e "$CINESCALE_REPO/models/Wan-AI/Wan2.1-T2V-14B" ] || ln -s "$WAN_T2V_CKPT" "$CINESCALE_REPO/models/Wan-AI/Wan2.1-T2V-14B"
      # cinescale_t2v14b_ours.py hard-codes 50 steps and seed 0 (the paper's protocol).
      { [ "$STEPS" = 50 ] && [ "$SEED" = 0 ]; } || warn "CineScale ignores STEPS=$STEPS / SEED=$SEED; it always uses 50 steps and seed 0"
      emit_script "$out/cmd.sh" "$CINESCALE_REPO" "$out/run_log.log" \
        CUDA_VISIBLE_DEVICES="$gpus" CS_LORA=0 CS_SKIP_DOWNLOAD=1 CS_OUT="$out/video.mp4" CS_PROMPT="$p" \
        CS_FRAMES="$FRAMES" CS_HEIGHT=$((H / U)) CS_WIDTH=$((W / U)) CS_TARGET_H="$H" CS_TARGET_W="$W" -- \
        "$CINESCALE_ENV/bin/torchrun" --nproc_per_node=4 --master_port "$(free_port)" cinescale_t2v14b_ours.py ;;
    freeswim)
      n=1; gpus="$(gpus_for 1)"
      [ -f "$FREESWIM_REPO/inference_14b_ours.py" ] || die "FreeSwim driver not found in $FREESWIM_REPO (run env/setup_envs.sh freeswim)"
      # The 14B model, explicitly: the script's default is the 1.3B model, and comparing a 1.3B
      # baseline with a 14B HyperGen would measure model size, not method.
      emit_script "$out/cmd.sh" "$FREESWIM_REPO" "$out/run_log.log" \
        CUDA_VISIBLE_DEVICES="$gpus" FREESWIM_MODEL_ID="$WAN_T2V_DIFFUSERS_CKPT" FREESWIM_OUT="$out/video.mp4" \
        FREESWIM_PROMPT="$p" FREESWIM_BASE_H=720 FREESWIM_BASE_W=1280 FREESWIM_FRAMES="$FRAMES" \
        FREESWIM_SEED="$SEED" -- \
        "$FREESWIM_ENV/bin/python" inference_14b_ours.py --target_height "$H" --target_width "$W" ;;
  esac
  RUN_GPUS="$gpus"
  if [ "$DRY" = 1 ]; then log "DRY  $m $res ${n}GPU :: $p"; return 0; fi
  log "run   $m $res ${n}GPU :: $p"
  local started=$(( $(date +%s) - 2 )) t0=$SECONDS rc note=""
  bash "$out/cmd.sh"; rc=$?
  RUN_WALL=$((SECONDS - t0))
  RUN_VIDEO="$(find_video "$out" "$started")"
  if [ -z "$RUN_VIDEO" ]; then
    [ "$rc" -eq 0 ] && rc=90
    if grep -q "OutOfMemoryError\|CUDA out of memory" "$out/run_log.log" 2>/dev/null; then note="out of memory"; fi
    warn "no video from $m $res (rc=$rc${note:+, $note})"
  fi
  local coll=0 hr
  [ "$rc" -eq 0 ] && collect T1 "$key" "$out" "{\"table\": 3, \"baseline\": \"$m\", \"res\": \"$res\"}" && coll=1
  hr="$(highres_seconds "$out/run_log.log")"
  tsv_row "$T" "$m" "$res" "$p" "$(hostname)" "$gpus" "$RUN_WALL" "$rc" "$coll" "$(gpu_clocks "$gpus")" "${hr:-}" "$note"
  log "      rc=$rc  high-res stage=${hr:-n/a}s  wall=${RUN_WALL}s"
}

if [ "$SKIP_GEN" = 0 ]; then
  # FreeSwim is single-GPU and takes ~3.5 h per 2K video, so with 5+ GPUs its videos run in the
  # background, one per GPU from offset 4 on, while CineScale uses GPUs 0-3. (Separate GPUs do not
  # slow each other; each H100 enforces its own power limit.)
  NG="$(n_gpus_available)"; fg_jobs=(); fs_jobs=(); pids=()
  for item in "${PLAN[@]}"; do
    IFS='|' read -r m res n <<<"$item"
    mapfile -t ps < <(prompts_for "$res" "$n")
    for p in "${ps[@]}"; do
      if [ "$m" = freeswim ] && [ "$NG" -ge 5 ]; then fs_jobs+=("$res|$p"); else fg_jobs+=("$m|$res|$p"); fi
    done
  done
  if [ "${#fs_jobs[@]}" -gt 0 ]; then
    nw=$((NG - 4)); [ "$nw" -gt "${#fs_jobs[@]}" ] && nw="${#fs_jobs[@]}"
    for ((w = 0; w < nw; w++)); do
      ( for ((j = w; j < ${#fs_jobs[@]}; j += nw)); do
          IFS='|' read -r res p <<<"${fs_jobs[$j]}"
          GPU_OFFSET=$((4 + w)) measure_baseline freeswim "$res" "$p"
        done ) > "$RESULTS/T5_algo/freeswim_worker$w.log" 2>&1 &
      pids+=($!)
    done
    log "FreeSwim: ${#fs_jobs[@]} video(s) on $nw background GPU(s); logs in $RESULTS/T5_algo/freeswim_worker*.log"
  fi
  for job in "${fg_jobs[@]}"; do
    IFS='|' read -r m res p <<<"$job"
    measure_baseline "$m" "$res" "$p"
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
fi
[ "$DRY" = 1 ] && exit 0
[ "$SKIP_SCORE" = 1 ] && exit 0

VB="$RESULTS/T1_quality/vbench.tsv"
for item in "${PLAN[@]}"; do
  IFS='|' read -r m res n <<<"$item"
  vids=()
  while IFS= read -r p; do
    f="$WORK/videos/T1/${m}__${res}__baseline__$(slugify "$p").mp4"
    [ -s "$f" ] && vids+=("$f")
  done < <(prompts_for "$res" "$n")
  [ "${#vids[@]}" -eq 0 ] && continue
  vbench_cell "$VB" "config=$m res=$res variant=baseline" "${vids[@]}" || warn "VBench failed for $m $res"
done
"$METRICS_ENV/bin/python" "$AE_ROOT/eval/build_latency.py" --results "$RESULTS"
log "done. Next: bash plots/make_all.sh --results $RESULTS"
