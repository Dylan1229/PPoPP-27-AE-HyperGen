#!/usr/bin/env bash
# Figure 12: Stage-2 latency of CogVideoX at 4K, adding one design point at a time. Each step adds
# exactly one mechanism to the previous one, so the increments are attributable:
#   baseline      1 GPU, no cache
#   tilepar       + tile parallelism on 4 GPUs
#   cache         + the inter-step / intra-tile cache with ONE uniform threshold (per-tile scale 0)
#   regionaware   + region-aware cache control (per-tile threshold, scale factor 0.5)
#   rebalance     + cache-guided workload rebalancing (ENABLE_REDISTRIBUTE=1)
# Stage 1 and the upsampling are identical in all five and excluded, as in the paper.
#
#   bash scripts/run_fig12.sh --quick   the four 4-GPU steps (~30 min)
#   bash scripts/run_fig12.sh --full    all five, including the 1-GPU baseline (~50 min)
# Options: --dry-run
#
# Output:  $RESULTS/T7_figures/timings.tsv  rows with block=fig12
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"
banner "Figure 12: technique breakdown (CogVideoX 4K)"
require_gpus 4

CFG="${MODELS_OVERRIDE:-cog}"; P="$FIG_PROMPT_4K"; N=4
# tilepar runs first: it writes the shared Stage-1 latent that every later step loads.
measure_t7 fig12 "$CFG" 4k tilepar     "$N" "$P" nocache ENABLE_REDISTRIBUTE=0
[ "$MODE" = full ] && measure_t7 fig12 "$CFG" 4k baseline 1 "$P" nocache ENABLE_REDISTRIBUTE=0
measure_t7 fig12 "$CFG" 4k cache       "$N" "$P" plain   ENABLE_REDISTRIBUTE=0
measure_t7 fig12 "$CFG" 4k regionaware "$N" "$P" ours    ENABLE_REDISTRIBUTE=0
measure_t7 fig12 "$CFG" 4k rebalance   "$N" "$P" ours    ENABLE_REDISTRIBUTE=1
log "done. Next: bash plots/make_all.sh --results $RESULTS"
