#!/usr/bin/env bash
# Table 5: HyperGen's cache policy against TeaCache, both running on HyperGen's tiles in the same
# CogVideoX-1.5 pipeline, at 2K on ONE GPU per run (so tile parallelism is not a factor). Each
# method is measured at its conservative ("slow") and aggressive ("fast") operating point;
# speedup and retention are against the no-cache run of the same pipeline, seed and Stage-1 latent.
#
#   bash scripts/run_table5.sh --quick   the five rows of Table 5 (~15 min, 5 GPUs)
#   bash scripts/run_table5.sh --full    adds the third point of each sweep (ours 0.30, TeaCache 0.30)
#                                        reported in the supplementary material (~15 min, 7 GPUs)
# Options: --dry-run --score-only --no-score
#
# The reference runs first and alone, because every other run loads its Stage-1 latent; the cache
# runs then go side by side, one GPU each (as in the paper's measurement). Set SEQUENTIAL=1 to run
# them one after another on a single GPU instead (~25 min).
#
# Output:  $RESULTS/T2_baselines/timings.tsv   $RESULTS/T2_baselines/quality.tsv
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"

CELLS=("ours|0.15" "ours|0.20" "teacache|0.10" "teacache|0.20")   # slow / fast of each method
[ "$MODE" = full ] && CELLS+=("ours|0.30" "teacache|0.30")
banner "Table 5: HyperGen cache vs TeaCache (CogVideoX 2K, 1 GPU)"
[ "${SEQUENTIAL:-0}" = 1 ] || require_gpus $((1 + ${#CELLS[@]}))

CFG=cog; RES=2k; PROMPT="$FIG_PROMPT_2K"
T="$RESULTS/T2_baselines/timings.tsv"
tsv_init "$T" config res method thresh host gpu wall_s stage1_s upscale_s stage2_s skip_rate rc collected clocks

run_cell() {   # method thresh gpu-offset
  local m="$1" th="$2" off="$3"
  local key="${CFG}__${RES}__${m}__${th}"
  if already_done T2 "$key"; then log "skip  $key (already measured)"; return 0; fi
  case "$m" in
    nocache)  set_cache_flags "$CFG" nocache ;;
    ours)     set_cache_flags "$CFG" "ours:$th" ;;
    teacache) set_cache_flags "$CFG" "teacache:$th" ;;
  esac
  local out="$WORK/runs/T2/$key"
  log "run   $m thr=$th on GPU offset $off"
  GPU_OFFSET="$off" run_one "$CFG" "$RES" "$PROMPT" 1 "$out" ENABLE_REDISTRIBUTE=0
  [ "$DRY" = 1 ] && return 0
  local st coll=0
  st="$(stage_times "$out")"
  collect T2 "$key" "$out" "{\"table\": 5, \"method\": \"$m\", \"threshold\": \"$th\"}" && coll=1
  tsv_row "$T" "$CFG" "$RES" "$m" "$th" "$(hostname)" "$RUN_GPUS" "$RUN_WALL" "$(cut -f1 <<<"$st")" \
    "$(cut -f3 <<<"$st")" "$(cut -f2 <<<"$st")" "$(skip_rate "$out")" "$RUN_RC" "$coll" "$(gpu_clocks "$RUN_GPUS")"
  log "      $m thr=$th rc=$RUN_RC stage2=$(cut -f2 <<<"$st")s skip=$(skip_rate "$out")"
}

if [ "$SKIP_GEN" = 0 ]; then
  run_cell nocache 0 0
  pids=(); off=1
  for c in "${CELLS[@]}"; do
    IFS='|' read -r m th <<<"$c"
    if [ "${SEQUENTIAL:-0}" = 1 ]; then
      run_cell "$m" "$th" 0
    else
      ( run_cell "$m" "$th" "$off" ) > "$RESULTS/T2_baselines/${m}_${th}.log" 2>&1 &
      pids+=($!); off=$((off + 1)); sleep 5
    fi
  done
  for pid in "${pids[@]}"; do wait "$pid"; done
  [ "${#pids[@]}" -gt 0 ] && cat "$RESULTS/T2_baselines/"*_*.log 2>/dev/null | grep -E '^\[' || true
fi
[ "$DRY" = 1 ] && exit 0
[ "$SKIP_SCORE" = 1 ] && exit 0

# ---------------------------------------------------------------- retention against the reference
Q="$RESULTS/T2_baselines/quality.tsv"
tsv_init "$Q" method thresh skip_rate stage2_s speedup psnr ssim lpips vbench
ref="$WORK/videos/T2/${CFG}__${RES}__nocache__0.mp4"
[ -s "$ref" ] || die "no no-cache reference video at $ref"
base="$(awk -F'\t' 'NR>1 && $3=="nocache" && $12==0 {s=$10} END {print s}' "$T")"
for c in "${CELLS[@]}"; do
  IFS='|' read -r m th <<<"$c"
  v="$WORK/videos/T2/${CFG}__${RES}__${m}__${th}.mp4"
  [ -s "$v" ] || { warn "missing $v"; continue; }
  scored_already "$Q" 8 "$m" "$th" && continue
  row="$(awk -F'\t' -v m="$m" -v t="$th" 'NR>1 && $3==m && $4==t && $12==0 {r=$10"\t"$11} END {print r}' "$T")"
  s2="$(cut -f1 <<<"$row")"; sk="$(cut -f2 <<<"$row")"
  sp="$(awk -v b="$base" -v s="$s2" 'BEGIN { if (s > 0 && b > 0) printf "%.2f", b / s }')"
  mt="$(retention "$ref" "$v")"
  tsv_row "$Q" "$m" "$th" "$sk" "$s2" "$sp" "$(cut -d, -f1 <<<"$mt")" "$(cut -d, -f2 <<<"$mt")" "$(cut -d, -f3 <<<"$mt")" ""
  log "$m thr=$th  speedup=${sp}x  psnr,ssim,lpips=${mt:-FAILED}"
done
log "done. Next: bash plots/make_all.sh --results $RESULTS"
