#!/usr/bin/env bash
# GPUs 4-7: FreeSwim env on a uv-managed Python (headers for Triton), fix the CineScale high-res
# timer row, then rerun the FreeSwim 2K quick video.
cd ~/ae/PPoPP-27-AE-HyperGen
export CKPT_ROOT=$HOME/ckpts GPU_LIST=4,5,6,7 PATH=$HOME/.local/bin:$PATH
L=~/ae/logs
bash env/setup_envs.sh freeswim > $L/setup_freeswim2.log 2>&1; echo "rc=$?" >> $L/setup_freeswim2.log
CUDA_VISIBLE_DEVICES=4 envs/freeswim/bin/python - >> $L/setup_freeswim2.log 2>&1 <<'PY'
import torch, sysconfig
f = torch.compile(lambda x: x.sin() + 1)
print("python include:", sysconfig.get_paths()["include"])
print("torch.compile smoke:", f(torch.ones(4, device="cuda")).sum().item())
PY
T=results/ae/T5_algo/timings.tsv
URS=35; eval "$(sed -n '/^highres_seconds()/,/^}/p' scripts/run_table3_baselines.sh)"
hr=$(highres_seconds work/runs/T5/cinescale__2k__baseline__a_person_is_putting_sauce_on_a_burger/run_log.log)
echo "cinescale 2k high-res stage re-parsed: ${hr}s" >> $L/setup_freeswim2.log
awk -F'\t' -v OFS='\t' -v hr="$hr" 'NR == 1 {print; next}
  $1 == "freeswim" && $7 != 0 {next}
  $1 == "cinescale" && $2 == "2k" && $7 == 0 {$10 = hr} {print}' $T > $T.tmp && mv $T.tmp $T
mv work/runs/T5/freeswim__2k__baseline__a_person_is_putting_sauce_on_a_burger work/runs/T5/_failed_freeswim_no_python_headers
bash scripts/run_table3_baselines.sh --quick --with-freeswim > $L/table3_baselines_quick3.log 2>&1
date -u > $L/groupD.done
