<div align="center">
<h1>
Denoise to Track: Harnessing Video Diffusion Priors for Robust Correspondence</h1>

[**Tianyu Yuan**](https://openreview.net/profile?id=~Tianyu_Yuan1)<sup>1</sup>, 
[**Yuanbo Yang**](https://freemty.github.io/)<sup>2</sup>, 
[**Lin-Zhuo Chen**](https://linzhuo.xyz/)<sup>1</sup>, 
[**Yao Yao**](https://yoyo000.github.io/)<sup>&dagger;1</sup>, 
[**Zhuzhong Qian**](https://cs.nju.edu.cn/qzz/)<sup>&dagger;1</sup>,

<sup>1</sup>NanJing University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>ZheJiang University

<sup>&dagger;</sup>Co-corresponding author.


<!-- <a href="https://arxiv.org/abs/2506.08015">
  <img src="https://img.shields.io/badge/2506.08015-arXiv-red" alt="arXiv">
</a>&emsp;&emsp;&emsp;&emsp; -->
<a href="https://yuantianyu177.github.io/heft_page/">
  <img src="https://img.shields.io/badge/HeFT-project_page-blue" alt="Project Page">
</a>
</div>


<!-- Please cite this paper if you find this repository useful.

```bash
@article{yuan2025denoise,
    title     = {Denoise to Track: Harnessing Video Diffusion Priors for Robust Correspondence},
    author    = {Yuan, Tianyu and Yang, Yuanbo and Chen, Lin-Zhuo and Yao, Yao and Qian, Zhuzhong},
    journal   = {arXiv preprint arXiv:2506.08015},
    year      = {2025}
}
``` -->

## Installation

```bash
conda create -n heft python=3.11 && conda activate heft
cd diffusers && pip install -e ".[torch]" && cd ..
pip install -e .
```

## Pretrained Model

We default to downloading the [Cosmos2](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) model, which requires NVIDIA approval and Hugging Face account login. You can find the pretrained model from [Hugging Face](https://huggingface.co/nvidia/Cosmos-Predict2-2B-Video2World) and download manually via:

```bash
python scripts/download_model.py
```

You can also skip this step and it will automatically download it when executing the following commands. **We strongly recommend using a single GPU for the first run.**
## TapVid Datasets

Download and extract the TapVid-DAVIS dataset:

```bash
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip
```
For TAP-Vid-Kinetics, please refer to the [TAP-Vid GitHub](https://github.com/google-deepmind/tapnet/tree/main/tapnet/tapvid).
## Evaluation

We provide support for several video backbone models: Wan2.1-T2V-1.3B, Cosmos-Predict2-2B-Video2World, CogVideoX-2B. We provide an extension interface where you can inherit from [`AttentionHook`](src/heft/attention_hook/base_attention_hook.py) to implement support for other models. Additionally, you need to add configurations in the [`generation.py`](src/heft/utils/generation.py) and modify the corresponding model's pipeline function. For reference, please see the [Cosmos pipeline implementation](diffusers/src/diffusers/pipelines/cosmos/pipeline_cosmos2_video2world.py).

### Extract Features from Diffusion Model

First, extract video frames from the dataset and generate diffusion features:

```bash
# Extract video from .pkl file
python scripts/extract_tapvid_video.py --dataset-dir tapvid_davis --dataset davis

# Get video path for generation
python scripts/get_video_path.py --video-dir datasets

# Generate diffusion features
python scripts/generate_video.py \
    --video_file video_path.txt \
    --start_step 34 \
    --model cosmos2 \
    --num_inference_steps 35 \
    --save_steps 34 \
    --save_layers 18 \
    --save_qk \
    --resolution 704 1280 \
    --gpu 0 \
    --save_dir output/tapvid_davis_cosmos2
```

**Parameters:**
- `--video_file`: Path to a text file containing video paths (one per line)
- `--start_step`: Starting diffusion step for feature extraction
- `--model`: Model name (cosmos2, wan2.1, or cogvideox)
- `--num_inference_steps`: Total number of denoising steps
- `--save_steps`: Diffusion steps at which to save features (can specify multiple)
- `--save_layers`: Transformer layers from which to extract features (can specify multiple)
- `--save_qk`: Flag to save query and key descriptors
- `--resolution`: Generation resolution [height, width]
- `--gpu`: GPU IDs to use (space-separated)
- `--save_dir`: Directory to save extracted features

Note that `--resolution`, `--start_step`, and `--num_inference_steps` are coupled with the underlying diffusion model. By default, we use the model's native resolution and default inference steps, and extract features at the last denoising step.
### Point Tracking

Run point tracking evaluation on TapVid-DAVIS:

```bash
./experiments/eval_davis.sh
```

This script performs point tracking evaluation by:
1. Extracting trajectories using diffusion model features via `get_trajectory.py`
2. Evaluating tracking accuracy against ground truth via `evaluate.py`

**Configurable parameters** (in `configs/cosmos_config.sh`):
- `step`: Diffusion step(s) to use for tracking
- `layer`: Transformer layer(s) to extract features from
- `head`: Attention head(s) to use (-1 for layer-level tracking)
- `query_feature_type`: Type of query features (query, key, or hidden_states)
- `target_feature_type`: Type of target features for matching
- `update_feature_type`: Type of features to update during tracking
- `freq_range`: Frequency range for feature filtering [min, max]
- `argmax_radius`: Radius for argmax search in feature matching
- `search_radius`: Search radius for trajectory tracking
- `vis_threshold`: Visibility threshold for occlusion detection
- `track_save_dir`: Directory to save predicted trajectories
- `eval_save_dir`: Directory to save evaluation metrics
- `feature_save_dir`: Directory containing precomputed diffusion features

### Dense Tracking

For dense tracking evaluation, use the dense tracking script:

```bash
./experiments/dense_tracking.sh
```

This script performs dense tracking by generating trajectories for a dense grid of query points across video frames. **Important**: If you want to filter query points using a mask, place the mask files at `{task_dir}/mask/{frame_id}.png` in your feature save directory for **all frames**. Each mask should be a binary image where white pixels indicate valid regions for tracking. If no masks are provided, dense tracking will use all grid points.

## License

This implementation is licensed under the Creative Commons license, as found in the LICENSE file.

The work built in this repository benefits from the following open-source projects:
* [Diffusers](https://github.com/huggingface/diffusers): Apache 2.0
