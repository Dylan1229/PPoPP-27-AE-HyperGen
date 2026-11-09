#!/usr/bin/env bash
# Figure 14: cache-guided workload rebalancing. 4K (9 tiles, which never divide evenly over 2/4/8
# GPUs) with the cache on, with and without rebalancing.
#
#   bash scripts/run_fig14.sh --quick   CogVideoX on 4 GPUs, w/o and w/ rebalance (~15 min)
#   bash scripts/run_fig14.sh --full    CogVideoX and HunyuanVideo on 2/4/8 GPUs (~3 h, 8 GPUs)
# Options: --models "cog hunyuan"  --dry-run
#
# Output:  $RESULTS/T7_figures/timings.tsv  rows with block=fig14 (variant cache_reb0 / cache_reb1)
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"
case "$MODE" in
  quick) MODELS="${MODELS_OVERRIDE:-cog}";         COUNTS="4" ;;
  full)  MODELS="${MODELS_OVERRIDE:-cog hunyuan}"; COUNTS="2 4 8" ;;
esac
banner "Figure 14: workload rebalance (4K, cache on)"
[ "$MODE" = full ] && require_gpus 8 || require_gpus 4

for cfg in $MODELS; do
  for n in $COUNTS; do
    for rb in 0 1; do
      measure_t7 fig14 "$cfg" 4k "cache_reb${rb}" "$n" "$FIG_PROMPT_4K" ours ENABLE_REDISTRIBUTE=$rb
    done
  done
done
log "done. Next: bash plots/make_all.sh --results $RESULTS"
