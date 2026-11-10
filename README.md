# HeFT

HeFT tracks points with intermediate attention features from video diffusion models. It can capture conditional post-RoPE Q/K and per-head V at a selected denoising step. Feature extraction, tracking, and dataset evaluation are independent.

## Setup

The project uses Python 3.12, uv, PyTorch with CUDA 12.8, and the vendored Diffusers 0.42.1 source tree. All backbones use this release with the project's video input and attention capture extensions.

```bash
uv sync --group dev
```

## DAVIS workflow

Download the official TAP-Vid DAVIS pickle:

```bash
mkdir -p datasets
curl -L https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip -o /tmp/tapvid_davis.zip
unzip /tmp/tapvid_davis.zip -d datasets
```

This creates `datasets/tapvid_davis/tapvid_davis.pkl`.

Machine-specific paths live in the ignored `.env` file:

```dotenv
HEFT_DATASET_ROOT=/path/to/tapvid_davis
HEFT_MODEL_CACHE_DIR=/path/to/model_cache
HEFT_MODEL=cosmos2
HEFT_TEMPORAL_MODE=post_dit
HEFT_FEATURE_OUTPUT_ROOT=/path/to/features/{model}_davis/{temporal_mode}
HEFT_EVALUATION_OUTPUT=/path/to/evaluations/{model}_davis_{temporal_mode}.json
HEFT_GPU_IDS=0,1,2,3,4,5,6,7
HEFT_OVERWRITE=false
HEFT_CAPTURE_VALUE=false
```

Choose a backbone with `HEFT_MODEL`. For checkpoints downloaded with ModelScope, set `HEFT_MODEL_PATH` to the local Diffusers directory containing `model_index.json`.

| `HEFT_MODEL` | Checkpoint | Default feature times for a 25-frame input |
| --- | --- | --- |
| `wan` | Wan 2.1 T2V 1.3B | 7, native VAE compression |
| `wan2.2-ti2v-5b` | Wan 2.2 TI2V 5B | 7, native VAE compression |
| `cogvideox` | CogVideoX 2B | 7, native VAE compression |
| `cogvideox1.5-5b` | CogVideoX 1.5 5B T2V | 4, native VAE compression followed by two-latent DiT patches |
| `cosmos2` | Cosmos Predict2 2B Video2World | 7, native VAE compression |
| `cosmos3-edge` | Cosmos3-Edge | 7, native VAE compression and the generation attention pathway |

All six backbones support the following values of `HEFT_TEMPORAL_MODE`. All modes use the posterior mean and the selected checkpoint's denoising schedule.

| Mode | Input video encoding and descriptor readout |
| --- | --- |
| `framewise` | Encode each RGB frame independently, then pass the sequence to the DiT. |
| `stride1` | Disable VAE temporal downsampling and encode the entire clip. |
| `pre_dit` | Encode with native compression, then interpolate latents to RGB times before the DiT. |
| `post_dit` (default) | Run the VAE and DiT at their native temporal resolutions. Save the sparse descriptors and interpolate them at tracking time. |
| `four_phase` | Encode four offset clips with native compression and run the DiT on each. Interleave descriptors at their RGB times and average the shared first frame. |

The codec or DiT receives repeat padding where it needs a complete temporal block. CogVideoX 1.5 always groups two latents per DiT time patch. In `four_phase`, its patch descriptors are interpolated to the phase's latent times before interleaving. Select token descriptors such as Q/K/V for this mode, since each phase has its own attention matrix.

The feature manifest records the mode and feature timeline in RGB frames. Include `{temporal_mode}` in the feature and evaluation output paths to keep experiments separate. The new backbones start with layer 18, head 0, and full-band K matching.

Extract features:

```bash
uv run python scripts/extract_dataset_features.py
```

Set `HEFT_CAPTURE_VALUE=true` to save `value.safetensors` alongside Q/K. Use a fresh `HEFT_FEATURE_OUTPUT_ROOT` for the new extraction.

Track and evaluate:

```bash
uv run python scripts/evaluate_davis_features.py
```

Set `HEFT_TIME_CALIBRATION=adaptive` to estimate a signed advance for each video from its RGB frames and saved descriptors. The evaluation JSON includes the estimate and its evidence for every video. An optional `HEFT_TIME_PRIOR_FRAMES` supplies a fallback when neither motion cue has a reliable estimate. Include `{time_calibration}` in `HEFT_EVALUATION_OUTPUT` to separate nominal and calibrated scores.

## Python API

```python
from heft import (
    COSMOS_2,
    ExtractionConfig,
    ExtractionTask,
    TailFramePolicy,
    extract_feature,
)
from heft.attn_hook import CaptureSpec, FeatureKind

task = ExtractionTask(
    name="video-id",
    input_video=video,  # CPU uint8 [T, C, H, W]
    output_dir="features/video-id",
    seed=42,
)
config = ExtractionConfig(
    capture=CaptureSpec(
        step=34,
        layers=(18,),
        heads=(6,),
        features=(FeatureKind.QUERY, FeatureKind.KEY),
    ),
    chunk_size=25,
    tail_policy=TailFramePolicy.DISCARD,
    temporal_mode="post_dit",
)

result = extract_feature(
    task,
    model=COSMOS_2,
    config=config,
    gpu_ids=(0, 1),
    cache_dir="/path/to/model_cache",
)
```

Use `features=(FeatureKind.VALUE,)` to save only the value projection before attention weighting. For V–V tracking, set `query_feature`, `target_feature`, and `update_feature` to `FeatureKind.VALUE` in `TrackingConfig` and use the full channel range `(0.0, 1.0)`.

Tracking is exposed through `track_feature()` and `track_features()`. Query generators include `ExplicitQueries`, `GridQueries`, `MaskGridQueries`, and `AnnotationQueries`. Dataset-backed results can be evaluated with `evaluate_dataset()`.

Use `heft.calibration.estimate_feature_video_lag()` with the original RGB frames and selected descriptor, then set `TrackingTask.time_advance_frames` to `estimate.advance_frames`. A positive advance reads a later feature for each timestamp while retaining the first feature of each clip.

## Development

```bash
uv run pytest
uv run ruff check src scripts test
uv run pyright
```
