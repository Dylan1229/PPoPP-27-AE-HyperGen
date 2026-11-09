# Frozen evaluation set

| File | Prompts | Used for |
|---|---|---|
| `eval_2k_30.txt` | 30 | every 2K and 720p run (Table 3 averages the first 5; the latency experiments use the first) |
| `eval_4k_20.txt` | 20 | every 4K run (Table 3 averages the first 3; the latency experiments use the first) |
| `vbench2_i2v_30.json`, `vbench2_i2v_20.json` | | the same subsets with their VBench-I2V metadata |

**The prompt text is the conditioning image's file name**: `../inputs/2k_1440x2560/<prompt>.jpg` and
`../inputs/4k_2160x3840/<prompt>.jpg`. Prompts contain spaces, commas and apostrophes, so quote them.
Each set is stored only at the resolution it is used at. 720p runs use the 2K images, which each
pipeline resizes to its 1280x720 Stage-1 input.

Same prompt, image, seed (0), frame count (41), step counts (50 total, 35 in Stage 2), resolution and
GPU count across every method in a comparison. Only the method under test changes.

## How the subsets were selected

- **Source**: prompts were selected from `vbench2_i2v_full_info.json`.
- **Evaluated dimensions**: `background_consistency`, `aesthetic_quality`, `imaging_quality`, `motion_smoothness`, `subject_consistency`.
- **Goal**: create 10/20/30-prompt subsets that minimize spread across these five dimensions. Exact equality isn’t possible because `aesthetic_quality` and `imaging_quality` appear in both subject and background items, while `motion_smoothness`/`subject_consistency` only appear in subject items and `background_consistency` only in background items.
- **Method**: filter to the two relevant tag-combinations (subject-group and background-group), then split each subset evenly between them and shuffle. This yields the smallest possible spread across the five target dimensions.
- **Files**: `vbench2_i2v_10.json`, `vbench2_i2v_20.json`, `vbench2_i2v_30.json`.

### Per-dimension counts (only the evaluated five dimensions)

Counts below refer to each subset file.

- **vbench2_i2v_10.json** (10 prompts)
  - **background_consistency**: 5
  - **aesthetic_quality**: 10
  - **imaging_quality**: 10
  - **motion_smoothness**: 5
  - **subject_consistency**: 5

- **vbench2_i2v_20.json** (20 prompts)
  - **background_consistency**: 10
  - **aesthetic_quality**: 20
  - **imaging_quality**: 20
  - **motion_smoothness**: 10
  - **subject_consistency**: 10

- **vbench2_i2v_30.json** (30 prompts)
  - **background_consistency**: 15
  - **aesthetic_quality**: 30
  - **imaging_quality**: 30
  - **motion_smoothness**: 15
  - **subject_consistency**: 15


