# Vendored HyperGen implementation

This directory is a verbatim snapshot of the HyperGen implementation, maintained as the `PPoPP-dev`
branch of https://github.com/Dylan1229/SuperGen ("SuperGen" is the project's earlier name):

    commit a1bbe4b4e0305e411ab195fd6d8bd770134bdbcf  (2026-08-04)

Two files were removed, `input_image/` and `scripts/prompts/`, because the artifact's frozen
evaluation set in `../data/` supersedes them. The upstream license files of
HunyuanVideo-I2V (`HunyuanVideoI2V/LICENSE.txt`, `Notice`) were added; see
`../THIRD_PARTY_NOTICES.md`.

The paper's measurements ran at commits `bc3dc73` and `e7bd6bc` of the same branch, which are
ancestors of this snapshot. The commits in between change only the optional Wan2.1 T2V
sequence-parallel path and how the Wan2.1 runtime is located (`$WAN_REPO`).

Where things are:

| Component (paper section) | Code |
|---|---|
| Two-stage generation, tiling, tile shifting (Sec. 4) | `CogvideoI2V/pipeline_cogvideox_i2v_TVG.py`, `HunyuanVideoI2V/hyvideo/diffusion/pipelines/pipeline_hunyuan_video.py`, `WanI2V/tiled_stage2.py`, `utils/tile_utils.py` |
| Inter-step fine-grained cache, gain approximation (Sec. 5.1, Eq. 2-5) | `CogvideoI2V/modules/cogvideo_transformer_3d.py` (`check_skippable`), `HunyuanVideoI2V/hyvideo/modules/models.py`, `WanI2V/tiled_i2v.py` |
| Intra-step region-aware cache control (Sec. 5.2, Algorithm 2) | `utils/tile_std_tracker.py` |
| Tile parallelism and cache-guided workload rebalance (Sec. 6) | `utils/distributed.py` (`ENABLE_REDISTRIBUTE=1` turns rebalancing on) |
| Entry points | `CogvideoI2V/pipeline.py`, `HunyuanVideoI2V/sample_image2video.py`, `WanI2V/pipeline.py` (I2V), `WanI2V/pipeline_t2v.py` (T2V) |
