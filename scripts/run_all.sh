#!/usr/bin/env bash
# Run every main-paper experiment, then regenerate the tables and figures.
#
#   bash scripts/run_all.sh --quick    every experiment in its reduced configuration  (~3 h, 8 GPUs,
#                                      CogVideoX weights only; CineScale only if installed)
#   bash scripts/run_all.sh --full     the paper's configurations (several days on one 8xH100 node;
#                                      see README.md for per-experiment times)
#
# Each runner resumes where it stopped (finished runs are skipped), so this can be interrupted and
# restarted. Per-experiment logs go to $RESULTS/logs/.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
source "$HERE/lib.sh"
M="${1:---quick}"
case "$M" in --quick|--full) ;; *) die "usage: run_all.sh --quick|--full" ;; esac
mkdir -p "$RESULTS/logs"

run() {   # name script args...
  local name="$1"; shift
  log "==== $name"
  if bash "$HERE/$1" "${@:2}" 2>&1 | tee "$RESULTS/logs/$name.log"; then :; else warn "$name failed; see $RESULTS/logs/$name.log"; fi
}
run table4 run_table4.sh "$M"
run table5 run_table5.sh "$M"
run fig11  run_fig11.sh  "$M"
run fig12  run_fig12.sh  "$M"
run fig13  run_fig13.sh  "$M"
run fig14  run_fig14.sh  "$M"
run table3 run_table3.sh "$M"
if [ -f "$CINESCALE_REPO/cinescale_t2v14b_ours.py" ]; then
  run table3_baselines run_table3_baselines.sh "$M"
else
  warn "CineScale not installed (env/setup_envs.sh cinescale); skipping the Table 3 baseline rows"
fi
bash "$AE_ROOT/plots/make_all.sh" --results "$RESULTS" 2>&1 | tee "$RESULTS/logs/make_all.log"
