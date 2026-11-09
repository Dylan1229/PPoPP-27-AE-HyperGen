#!/usr/bin/env bash
# Getting started (~30 minutes on 4 H100s, after env setup and the CogVideoX download).
#
#   bash scripts/getting_started.sh
#
# 1. Pre-flight check of GPUs, environments and weights                          (~1 min)
# 2. Regenerate the paper's tables and figures from its raw data, no GPU          (~1 min)
# 3. One real measurement: CogVideoX-1.5 2K video on 4 GPUs, without and with
#    HyperGen's cache (the 2K / 4-GPU CogVideo row of Table 4)                    (~12 min)
# 4. Score it: PSNR/SSIM/LPIPS of the cached video against the uncached one,
#    and VBench of both (the first VBench call downloads its scoring models)      (~10 min)
# 5. Build Table 4 from this run and print it next to the paper's numbers         (~1 min)
#
# Expected (paper, Table 4, CogVideo 2K on 4 GPUs): Stage 2 61 s with cache vs 106 s without
# (1.74x), PSNR 37.08, SSIM 0.96, LPIPS 0.039. Absolute seconds vary with the machine; the speedup
# and the retention should be close.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/lib.sh"
MODE=quick
T0=$SECONDS
mkdir -p "$WORK" "$RESULTS"

echo "######## 1/5  pre-flight check"
bash "$HERE/check.sh" | tee "$WORK/check.log" 2>/dev/null || bash "$HERE/check.sh"
for k in "cogvideo  (torch" "metrics   (" "vbench    (" "CogVideoX1.5-5B-I2V"; do
  grep -F "$k" "$WORK/check.log" 2>/dev/null | grep -q '\[ ok \]' || die "required item missing: '$k' (see the check above)"
done

echo; echo "######## 2/5  tables and figures from the paper's raw data (no GPU)"
bash "$AE_ROOT/plots/make_all.sh" --results "$AE_ROOT/results/paper" > "$WORK/make_all_paper.log" 2>&1
grep -E '^(generated|skipped|tables)' "$WORK/make_all_paper.log"
echo "   (full log: $WORK/make_all_paper.log; compare out/paper/tables/*.tex with the PDF)"

echo; echo "######## 3-4/5  CogVideoX 2K on 4 GPUs, without and with the cache, then scoring"
T4_PLAN="2k 4" bash "$HERE/run_table4.sh" --quick

echo; echo "######## 5/5  Table 4 from this run"
bash "$AE_ROOT/plots/make_all.sh" --results "$RESULTS" > "$WORK/make_all_ae.log" 2>&1
awk '/^## Table 4/ {p = 1; print; next} /^## / {p = 0} p' "$OUT_ROOT/$(basename "$RESULTS")/summary.md"
echo
echo "generated table: $OUT_ROOT/$(basename "$RESULTS")/tables/table4_cache_speedup.tex"
echo "videos:          $WORK/videos/T7/"
echo "getting started finished in $(( (SECONDS - T0) / 60 )) min"
