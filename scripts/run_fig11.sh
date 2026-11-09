#!/usr/bin/env bash
# Figure 11: end-to-end latency (Stage 1 + upsampling + Stage 2) under three settings:
#   Origin                 1 GPU, no cache
#   +Parallelism           tile parallelism on 4 GPUs (2K) / 8 GPUs (4K), workload rebalance on
#   +Parallelism +Cache    the same plus HyperGen's cache at the backbone's operating point
#
#   bash scripts/run_fig11.sh --quick   CogVideoX, 2K: the three bars (~20 min, 4 GPUs)
#   bash scripts/run_fig11.sh --full    CogVideoX, HunyuanVideo, Wan2.1 I2V at 2K and 4K
#                                       (~6.5 h, 8 GPUs; the 1-GPU 4K Wan run alone takes ~80 min)
# Options: --models "cog hunyuan wan_i2v"  --dry-run
#
# Stage 1 runs on one GPU in every setting (no mechanism under test accelerates it), so it is timed
# once per backbone and resolution, in a run that generates a fresh sketch (variant "stage1"; only its
# Stage-1 timer is used), and the figure adds that time to every bar. The three bars themselves load
# the shared sketch, like every other comparison here: a run that generates Stage 1 in-process
# re-noises Stage 2 from a different random state, which changes the cache's decisions.
#
# Output:  $RESULTS/T7_figures/timings.tsv  rows with block=fig10
#          $RESULTS/T8_stage1/timings.tsv   the Stage-1 time per backbone used by the figure
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"

case "$MODE" in
  quick) MODELS="${MODELS_OVERRIDE:-cog}";                 PLAN=("2k 4") ;;
  full)  MODELS="${MODELS_OVERRIDE:-cog hunyuan wan_i2v}"; PLAN=("2k 4" "4k 8") ;;
esac
banner "Figure 11: end-to-end latency"
for cell in "${PLAN[@]}"; do read -r _ n <<<"$cell"; require_gpus "$n"; done

S1T="$RESULTS/T8_stage1/timings.tsv"
tsv_init "$S1T" config res host gpus wall_s stage1_s rc clocks
for cfg in $MODELS; do
  for cell in "${PLAN[@]}"; do
    read -r res n <<<"$cell"
    p="$FIG_PROMPT_2K"; [ "$res" = 4k ] && p="$FIG_PROMPT_4K"
    FRESH_STAGE1=1 measure_t7 fig10 "$cfg" "$res" stage1 "$n" "$p" nocache ENABLE_REDISTRIBUTE=0
    measure_t7 fig10 "$cfg" "$res" origin   1    "$p" nocache ENABLE_REDISTRIBUTE=0
    measure_t7 fig10 "$cfg" "$res" parallel "$n" "$p" nocache ENABLE_REDISTRIBUTE=1
    measure_t7 fig10 "$cfg" "$res" parcache "$n" "$p" ours    ENABLE_REDISTRIBUTE=1
  done
done
[ "$DRY" = 1 ] && exit 0

# Record each backbone's Stage-1 time from its fresh "stage1" runs (median over resolutions; it is
# the same 720p generation on rank 0 either way).
for cfg in $MODELS; do
  vals=$(awk -F'\t' -v c="$cfg" 'NR>1 && $1=="fig10" && $2==c && $4=="stage1" && $13==0 && $9!="" {print $9}' \
           "$RESULTS/T7_figures/timings.tsv" | sort -g)
  [ -z "$vals" ] && { warn "no Stage-1 timer for $cfg"; continue; }
  med=$(awk '{a[NR]=$1} END {print (NR % 2 ? a[(NR+1)/2] : (a[NR/2] + a[NR/2+1]) / 2)}' <<<"$vals")
  tsv_row "$S1T" "$cfg" 720p "$(hostname)" 0 "" "$med" 0 ""
  log "Stage 1 ($cfg): ${med}s (median of $(wc -l <<<"$vals") fresh runs)"
done
log "done. Next: bash plots/make_all.sh --results $RESULTS"
