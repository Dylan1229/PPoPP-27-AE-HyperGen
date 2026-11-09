#!/usr/bin/env bash
# 2K quick experiments on GPUs 0-3, after the getting-started test, then one run per other backbone.
cd ~/ae/PPoPP-27-AE-HyperGen
export CKPT_ROOT=$HOME/ckpts GPU_LIST=0,1,2,3
L=~/ae/logs
until grep -q "getting started finished" $L/getting_started.log; do sleep 30; done
bash scripts/run_table4.sh --quick       > $L/table4_quick.log 2>&1
bash scripts/run_fig11.sh --quick        > $L/fig11_quick.log 2>&1
bash scripts/run_fig13.sh --quick        > $L/fig13_quick.log 2>&1
bash scripts/run_table3.sh --quick       > $L/table3_quick.log 2>&1
SEQUENTIAL=1 bash scripts/run_table5.sh --quick > $L/table5_quick.log 2>&1
date -u > $L/groupA_cog.done
# every other backbone once (needs the wan env and the Wan I2V / Hunyuan weights)
until grep -q "rc=" $L/setup_wan2.log 2>/dev/null && grep -q DL_DONE $L/download.log; do sleep 60; done
T4_PLAN="2k 4" bash scripts/run_table4.sh --quick --models "hunyuan wan_i2v" > $L/table4_backbones_quick.log 2>&1
bash scripts/run_table3.sh --quick --models wan_t2v > $L/table3_want2v_quick.log 2>&1
date -u > $L/groupA.done
