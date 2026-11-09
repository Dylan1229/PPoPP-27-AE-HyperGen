#!/usr/bin/env bash
# Table 4: Stage-2 latency with and without HyperGen's cache, the speedup, and the visual retention
# of the cached output (PSNR / SSIM / LPIPS against the NO-CACHE output of the same pipeline and
# seed), plus VBench of both videos. Workload rebalancing is disabled, per the table's caption.
#
#   bash scripts/run_table4.sh --quick    CogVideoX, 2K: the 4-GPU and 1-GPU pairs (~25 min, 4 GPUs)
#   bash scripts/run_table4.sh --full     CogVideoX, HunyuanVideo, Wan2.1 I2V; 2K on 1 and 4 GPUs,
#                                         4K on 4 GPUs (~7 h, 4 GPUs)
#
# Options: --models "cog hunyuan wan_i2v"  --dry-run  --score-only  --no-score
#
# Output:  $RESULTS/T7_figures/timings.tsv        rows with block=fig11
#          $RESULTS/T7_figures/fig11_quality.tsv  PSNR/SSIM/LPIPS of each cached run vs its reference
#          $RESULTS/T7_figures/fig11_vbench.tsv   VBench of every video (one prompt per cell)
# Then:    bash plots/make_all.sh --results "$RESULTS"   -> out/<name>/tables/table4_cache_speedup.tex
set -uo pipefail
source "$(dirname "$0")/lib.sh"
parse_common_args "$@"

case "$MODE" in
  quick) MODELS="${MODELS_OVERRIDE:-cog}";                 PLAN=("2k 4" "2k 1") ;;
  full)  MODELS="${MODELS_OVERRIDE:-cog hunyuan wan_i2v}"; PLAN=("2k 4" "4k 4" "2k 1") ;;
esac
# T4_PLAN="2k 4;4k 4" overrides the (resolution, GPU count) cells; getting_started.sh uses "2k 4".
[ -n "${T4_PLAN:-}" ] && IFS=';' read -r -a PLAN <<<"$T4_PLAN"
banner "Table 4: cache speedup and retention"
require_gpus 4

prompt_for() {   # config res
  if [ "$2" = 4k ]; then echo "$FIG_PROMPT_4K"
  elif [ "$1" = hunyuan ]; then echo "$FIG_PROMPT_2K_HUNYUAN"
  else echo "$FIG_PROMPT_2K"; fi
}

# ---------------------------------------------------------------- generate
if [ "$SKIP_GEN" = 0 ]; then
  for cfg in $MODELS; do
    for cell in "${PLAN[@]}"; do
      read -r res n <<<"$cell"
      p="$(prompt_for "$cfg" "$res")"
      # No-cache FIRST: it generates the Stage-1 latent that the cached run then loads, so the two
      # differ only in Stage 2.
      measure_t7 fig11 "$cfg" "$res" nocache "$n" "$p" nocache ENABLE_REDISTRIBUTE=0
      measure_t7 fig11 "$cfg" "$res" cache   "$n" "$p" ours    ENABLE_REDISTRIBUTE=0
    done
  done
fi
[ "$DRY" = 1 ] && exit 0
[ "$SKIP_SCORE" = 1 ] && exit 0

# ---------------------------------------------------------------- score
QUAL="$RESULTS/T7_figures/fig11_quality.tsv"
VB="$RESULTS/T7_figures/fig11_vbench.tsv"
tsv_init "$QUAL" config res ngpu prompt psnr ssim lpips
V="$WORK/videos/T7"
for cfg in $MODELS; do
  for cell in "${PLAN[@]}"; do
    read -r res n <<<"$cell"
    slug="$(slugify "$(prompt_for "$cfg" "$res")" | cut -c1-40)"
    ref="$V/fig11__${cfg}__${res}__nocache__${n}gpu__${slug}.mp4"
    tst="$V/fig11__${cfg}__${res}__cache__${n}gpu__${slug}.mp4"
    if [ ! -s "$ref" ] || [ ! -s "$tst" ]; then warn "missing video(s) for $cfg $res ${n}GPU; not scored"; continue; fi
    if ! scored_already "$QUAL" 7 "$cfg" "$res" "$n"; then
      m="$(retention "$ref" "$tst")"
      tsv_row "$QUAL" "$cfg" "$res" "$n" "$slug" "$(cut -d, -f1 <<<"$m")" "$(cut -d, -f2 <<<"$m")" "$(cut -d, -f3 <<<"$m")"
      log "retention $cfg $res ${n}GPU: psnr,ssim,lpips = ${m:-FAILED}"
    fi
    for var in nocache cache; do
      f="$V/fig11__${cfg}__${res}__${var}__${n}gpu__${slug}.mp4"
      scored_already "$VB" 11 "$cfg" "$res" "$n" "$var" && continue
      vbench_cell "$VB" "config=$cfg res=$res ngpu=$n variant=$var" "$f" || warn "VBench failed: $f"
    done
  done
done
log "done. Next: bash plots/make_all.sh --results $RESULTS"
