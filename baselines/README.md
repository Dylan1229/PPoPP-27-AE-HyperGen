# Table 3 baselines: CineScale and FreeSwim

Both are training-free high-resolution video generators built on Wan2.1, so they run on the same
backbone (Wan2.1 T2V 14B), prompts, seed, frame count and resolutions as HyperGen's Wan2.1 T2V rows.
`env/setup_envs.sh cinescale freeswim` sets them up and `scripts/run_table3_baselines.sh` runs them.
The paper's runs used exactly the code below; `results/paper/manifests/{cinescale,freeswim}__*.json`
record each command.

Neither upstream entry point exposes the settings the comparison has to fix (prompt, frame count,
720p base resolution, 14B model, target resolution), so each is driven by a parameterised copy.

## CineScale (`baselines/cinescale/`, upstream Apache-2.0)

- **Upstream:** https://github.com/PeterZs/CineScale at `0abca0cfc3910b89fe2f7aa94ba6a9b0fc0714c7`
  ("fix setting", 2025-08-25), an ancestor of Eyeline-Labs/CineScale. Do not update it: Eyeline's
  current main is rewritten around Wan2.2 and no longer contains `diffsynth/`.
- **`cinescale_t2v14b_ours.py`:** upstream `cinescale_t2v14b_pro.py`, parameterised by environment
  variables: `CS_PROMPT`, `CS_FRAMES`, `CS_HEIGHT`/`CS_WIDTH` (base, 720x1280), `CS_TARGET_H`/`CS_TARGET_W`,
  `CS_OUT`. `CS_LORA=0` skips CineScale's trained LoRA. This is the training-free arm that Table 3
  reports; the default is 1, so the runner sets it explicitly. The script loads weights from
  `models/Wan-AI/Wan2.1-T2V-14B` inside the checkout (the runner links the download there). It
  hard-codes 50 steps and seed 0, which match the paper's protocol, so `STEPS` and `SEED` overrides
  do not reach CineScale.
- **`sdpa-scale-kwarg.patch`:** our only change to upstream code. Its PyTorch attention fallback
  passes FlashAttention's `softmax_scale=` to `F.scaled_dot_product_attention`, which names it
  `scale=`. A second assertion still requires `flash_attn`, so the env installs it.
- Runs on 4 GPUs with xDiT sequence parallelism (CineScale's own multi-GPU path).

## FreeSwim (`baselines/freeswim/`, upstream has no license file)

- **Upstream:** https://github.com/WillWu111/FreeSwim at `9f35a72f6382696fe2fd02ff02183a249c3090f3`
  (2025-12-02).
- **`freeswim-14b-ours.patch`:** because upstream has no license, the artifact ships our changes as a
  patch against upstream `inference.py` instead of a modified copy. `env/setup_envs.sh freeswim`
  applies it to create `third_party/FreeSwim/inference_14b_ours.py`, byte-identical to the file the
  paper's runs used. The patch changes three things. The model comes from `FREESWIM_MODEL_ID`;
  upstream defaults to the 1.3B model, and the runner passes the 14B Diffusers checkpoint. Prompt,
  frames (41), base resolution (720x1280, the 14B's training resolution) and seed (0) come from
  `FREESWIM_*` variables. Upstream's `assert target <= 1920x1088`, a guard for the 1.3B model,
  becomes a warning. The target resolution is passed as `--target_height/--target_width`.
- Single GPU with sequential CPU offload, as upstream runs it; its released code has no multi-GPU
  path. One 2K video takes ~3.5 h. At 4K it runs out of memory on an 80 GB H100, as the paper
  reports (`run_table3_baselines.sh --freeswim-4k` reproduces this).
