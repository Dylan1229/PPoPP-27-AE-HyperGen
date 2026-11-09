#!/usr/bin/env bash
# Figure 13: tile-parallel scalability of Stage 2: 2K on 1/2/4 GPUs and 4K on 1/2/4/8 GPUs, with the
# cache and workload rebalancing OFF so the curve isolates tile parallelism. (2K is not run beyond 4
# GPUs: it has only 4 tiles.)
#
#   bash scripts/run_fig13.sh --quick   CogVideoX 2K on 1/2/4 GPUs (~20 min, 4 GPUs)
#   bash scripts/run_fig13.sh --full    CogVideoX and HunyuanVideo, 2K and 4K (~4 h, 8 GPUs)
# Options: --models "cog hunyuan"  --dry-run
#
# Output:  $RESULTS/T7_figures/timings.tsv  rows with block=fig13
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"
case "$MODE" in
  quick) MODELS="${MODELS_OVERRIDE:-cog}";         PLAN=("2k 1 2 4") ;;
  full)  MODELS="${MODELS_OVERRIDE:-cog hunyuan}"; PLAN=("2k 1 2 4" "4k 1 2 4 8") ;;
esac
banner "Figure 13: tile-parallel scalability"
[ "$MODE" = full ] && require_gpus 8 || require_gpus 4

for cfg in $MODELS; do
  for cell in "${PLAN[@]}"; do
    read -r res counts <<<"$cell"
    p="$FIG_PROMPT_2K"; [ "$res" = 4k ] && p="$FIG_PROMPT_4K"
    for n in $counts; do
      measure_t7 fig13 "$cfg" "$res" nocache "$n" "$p" nocache ENABLE_REDISTRIBUTE=0
    done
  done
done
log "done. Next: bash plots/make_all.sh --results $RESULTS"
