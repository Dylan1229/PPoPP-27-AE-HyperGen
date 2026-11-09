# Third-party notices

The artifact's own code (`scripts/`, `eval/`, `plots/`, `env/`, and HyperGen's tiling, caching and
tile-parallel code in `hypergen/`) is released under the Apache License 2.0 (`LICENSE`). Parts of
`hypergen/` are derived from the upstream projects below and remain under their licenses.

| Path | Derived from | License |
|---|---|---|
| `hypergen/HunyuanVideoI2V/` (incl. `hyvideo/`) | [Tencent-Hunyuan/HunyuanVideo-I2V](https://github.com/Tencent-Hunyuan/HunyuanVideo-I2V) | Tencent Hunyuan Community License Agreement, see `hypergen/HunyuanVideoI2V/LICENSE.txt` and `Notice`. It does not apply in the EU, UK or South Korea. |
| `hypergen/CogvideoI2V/` | [huggingface/diffusers](https://github.com/huggingface/diffusers) CogVideoX pipeline and transformer; [THUDM/CogVideo](https://github.com/THUDM/CogVideo) | Apache-2.0 |
| `hypergen/CogvideoI2V/modules/teacache_*`, `adacache_*` | ports of [TeaCache](https://github.com/ali-vilab/TeaCache) and [AdaCache](https://github.com/AdaCache-DiT/AdaCache) onto HyperGen's tiles | Apache-2.0 |
| `hypergen/WanI2V/` | builds on [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1), which is cloned at setup time and not redistributed | Apache-2.0 |
| `eval/vbench_eval.py`, `eval/vbench_cells.py` | call [VBench](https://github.com/Vchitect/VBench) (cloned at setup time) | Apache-2.0 |
| `data/inputs/` | conditioning images and prompts from the VBench++ image-to-video suite (Huang et al., arXiv:2411.13503), redistributed at the 2560x1440 and 3840x2160 resolutions used in the paper, for evaluation only | see VBench |

**Model weights are not redistributed.** `env/download_models.sh` fetches them from Hugging Face,
and each is governed by its own license: the CogVideoX License (CogVideoX1.5-5B-I2V), the Tencent
Hunyuan Community License (HunyuanVideo-I2V), Apache-2.0 (Wan2.1), and the respective licenses of
the LLaVA-Llama-3 and CLIP text encoders.

The Table 3 baselines are cloned at setup time from the repositories pinned in `env/repos.txt` and
are governed by their own terms:

- `baselines/cinescale/cinescale_t2v14b_ours.py` and `sdpa-scale-kwarg.patch` are derived from
  [CineScale](https://github.com/PeterZs/CineScale) (Apache-2.0).
- [FreeSwim](https://github.com/WillWu111/FreeSwim) has no license file. The artifact therefore
  redistributes none of its code, only `baselines/freeswim/freeswim-14b-ours.patch`, our changes to
  its `inference.py`, which is applied to a fresh upstream checkout.
