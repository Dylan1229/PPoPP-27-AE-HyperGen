#!/usr/bin/env bash
# GPUs 4-7: CineScale env with the xfuser pin, Figure 11 with the corrected protocol, then the baselines.
cd ~/ae/PPoPP-27-AE-HyperGen
export CKPT_ROOT=$HOME/ckpts GPU_LIST=4,5,6,7
L=~/ae/logs
FORCE=1 bash env/setup_envs.sh cinescale > $L/setup_cinescale2.log 2>&1; echo "rc=$?" >> $L/setup_cinescale2.log
FORCE=1 bash scripts/run_fig11.sh --quick > $L/fig11_quick2.log 2>&1
rm -f results/ae/T5_algo/timings.tsv   # the failed CineScale attempt (xfuser 0.7)
bash scripts/run_table3_baselines.sh --quick --with-freeswim > $L/table3_baselines_quick2.log 2>&1
date -u > $L/groupC.done
