#!/usr/bin/env bash
# Table 3 (HyperGen rows): VBench quality of HyperGen at 720p / 2K / 4K, with and without the cache,
# and the Stage-2 latency column. Every run uses 4 GPUs with rebalancing off; the w/o-cache run of a
# prompt goes first and the cached run reuses its Stage-1 latent.
#
#   bash scripts/run_table3.sh --quick   CogVideoX, 2K, first prompt, w/o and w/ cache (~15 min)
#   bash scripts/run_table3.sh --full    CogVideoX, HunyuanVideo, Wan2.1 T2V (the three backbones
#                                        Table 3 shows): 720p x5 prompts, 2K x5, 4K x3, w/o and w/
#                                        cache (~13 h on 4 GPUs)
#
# Options: --models "cog hunyuan wan_t2v wan_i2v"  (wan_i2v is measured in the paper's data but not
#          shown in Table 3)  --dry-run  --score-only  --no-score
#
# The CineScale and FreeSwim rows are produced by scripts/run_table3_baselines.sh.
#
# Output:  $RESULTS/T1_quality/timings.tsv  one row per video (stage timers)
#          $RESULTS/T1_quality/vbench.tsv   one row per cell: VBench V1-V5 and their average
#          $RESULTS/T5_algo/latency.tsv     the latency column (eval/build_latency.py)
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"

case "$MODE" in
  quick) MODELS="${MODELS_OVERRIDE:-cog}"; RESOLUTIONS="2k"; N2=1; N4=1; N7=1 ;;
  full)  MODELS="${MODELS_OVERRIDE:-cog hunyuan wan_t2v}"; RESOLUTIONS="720p 2k 4k"; N2=$N_2K; N4=$N_4K; N7=$N_720P ;;
esac
banner "Table 3: VBench quality (HyperGen rows)"
require_gpus 4
NGPU=4
T="$RESULTS/T1_quality/timings.tsv"
tsv_init "$T" config res variant prompt host gpus wall_s rc collected stage1_s stage2_s upscale_s

n_for() { case "$1" in 720p) echo "$N7" ;; 2k) echo "$N2" ;; 4k) echo "$N4" ;; esac; }
# 720p has no cache row in Table 3; Wan2.1 T2V has no 720p row (T2V at 720p IS the base model).
variants_for() {
  if [ "$1" = 720p ]; then echo nocache; else echo "nocache cache"; fi
}

measure_t1() {   # config res variant prompt
  local cfg="$1" res="$2" var="$3" p="$4"
  local key="${cfg}__${res}__${var}__$(slugify "$p")"
  if already_done T1 "$key"; then log "skip  $key (already measured)"; return 0; fi
  if [ "$var" = cache ]; then set_cache_flags "$cfg" ours; else set_cache_flags "$cfg" nocache; fi
  local out="$WORK/runs/T1/$key"
  log "run   $cfg $res $var ${NGPU}GPU :: $p"
  run_one "$cfg" "$res" "$p" "$NGPU" "$out" ENABLE_REDISTRIBUTE=0
  [ "$DRY" = 1 ] && return 0
  local st coll=0
  st="$(stage_times "$out")"
  collect T1 "$key" "$out" "{\"table\": 3, \"config\": \"$cfg\", \"res\": \"$res\", \"variant\": \"$var\"}" && coll=1
  tsv_row "$T" "$cfg" "$res" "$var" "$p" "$(hostname)" "$RUN_GPUS" "$RUN_WALL" "$RUN_RC" "$coll" \
    "$(cut -f1 <<<"$st")" "$(cut -f2 <<<"$st")" "$(cut -f3 <<<"$st")"
  log "      rc=$RUN_RC  stage2=$(cut -f2 <<<"$st")s  wall=${RUN_WALL}s"
}

# ---------------------------------------------------------------- generate
if [ "$SKIP_GEN" = 0 ]; then
  for cfg in $MODELS; do
    for res in $RESOLUTIONS; do
      [ "$cfg" = wan_t2v ] && [ "$res" = 720p ] && continue
      while IFS= read -r -u 3 p; do
        for var in $(variants_for "$res"); do measure_t1 "$cfg" "$res" "$var" "$p"; done
      done 3< <(prompts_for "$res" "$(n_for "$res")")
    done
  done
fi
[ "$DRY" = 1 ] && exit 0
[ "$SKIP_SCORE" = 1 ] && exit 0

# ---------------------------------------------------------------- score
# One VBench run per cell over all of its prompts: Table 3 reports the mean over the prompt set.
VB="$RESULTS/T1_quality/vbench.tsv"
for cfg in $MODELS; do
  for res in $RESOLUTIONS; do
    [ "$cfg" = wan_t2v ] && [ "$res" = 720p ] && continue
    for var in $(variants_for "$res"); do
      vids=()
      while IFS= read -r p; do
        f="$WORK/videos/T1/${cfg}__${res}__${var}__$(slugify "$p").mp4"
        [ -s "$f" ] && vids+=("$f")
      done < <(prompts_for "$res" "$(n_for "$res")")
      [ "${#vids[@]}" -eq 0 ] && { warn "no videos for $cfg $res $var"; continue; }
      have=$(awk -F'\t' -v c="$cfg" -v r="$res" -v v="$var" \
        'NR>1 && $1==c && $2==r && $3==v && $10!="" {n=$4} END {print n+0}' "$VB" 2>/dev/null)
      if [ "${FORCE:-0}" != 1 ] && [ "${have:-0}" = "${#vids[@]}" ]; then
        log "skip  VBench $cfg $res $var (already scored over ${#vids[@]} video(s))"; continue
      fi
      vbench_cell "$VB" "config=$cfg res=$res variant=$var" "${vids[@]}" || warn "VBench failed for $cfg $res $var"
    done
  done
done
"$METRICS_ENV/bin/python" "$AE_ROOT/eval/build_latency.py" --results "$RESULTS"
log "done. Next: bash plots/make_all.sh --results $RESULTS"
