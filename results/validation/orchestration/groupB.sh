#!/usr/bin/env bash
# 4K quick experiments on GPUs 4-7 (no Stage-1 latent shared with the 2K runs on GPUs 0-3).
cd ~/ae/PPoPP-27-AE-HyperGen
export CKPT_ROOT=$HOME/ckpts GPU_LIST=4,5,6,7
L=~/ae/logs
bash scripts/run_fig12.sh --quick > $L/fig12_quick.log 2>&1
bash scripts/run_fig14.sh --quick > $L/fig14_quick.log 2>&1
until [ -f $L/all_envs.done ]; do sleep 30; done
bash scripts/run_table3_baselines.sh --quick > $L/table3_baselines_quick.log 2>&1
date -u > $L/groupB.done
