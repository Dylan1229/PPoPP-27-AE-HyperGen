#!/usr/bin/env bash
# Follow-ups after the GPU 0-3 batch: (1) re-measure the 4-GPU no-cache reference that generated its
# Stage 1 in-process (invalid for retention), (2) a fresh getting-started run that exercises the
# preparatory Stage-1 run.
cd ~/ae/PPoPP-27-AE-HyperGen
export CKPT_ROOT=$HOME/ckpts GPU_LIST=0,1,2,3
L=~/ae/logs
until [ -f $L/groupA.done ]; do sleep 60; done
K=fig11__cog__2k__nocache__4gpu__a_person_is_putting_sauce_on_a_burger
rm -f work/videos/T7/$K.mp4 results/ae/manifests/$K.json
awk -F"\t" "!(\$1==\"cog\" && \$2==\"2k\" && \$3==\"4\")" results/ae/T7_figures/fig11_quality.tsv > /tmp/q && mv /tmp/q results/ae/T7_figures/fig11_quality.tsv
awk -F"\t" "!(\$1==\"cog\" && \$2==\"2k\" && \$3==\"4\" && \$4==\"nocache\")" results/ae/T7_figures/fig11_vbench.tsv > /tmp/v && mv /tmp/v results/ae/T7_figures/fig11_vbench.tsv
bash scripts/run_table4.sh --quick > $L/table4_fixup.log 2>&1
until [ -f $L/groupB.done ]; do sleep 60; done
WORK=$HOME/ae/work_gs RESULTS=$HOME/ae/PPoPP-27-AE-HyperGen/results/gs OUT_ROOT=$HOME/ae/PPoPP-27-AE-HyperGen/out \
  bash scripts/getting_started.sh > $L/getting_started_fresh.log 2>&1
date -u > $L/groupA2.done
