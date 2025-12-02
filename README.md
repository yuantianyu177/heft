## Installation

```bash
conda create -n heft python=3.11 && conda activate heft
cd diffusers && pip install -e ".[torch]" && cd ..
pip install -e .
```

## Pretrained Model
```bash
python scripts/download_model.py
```


## TapVid Datasets
```bash
wget https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip
```


## Evaluate TapVid-DAVIS
### Extract features from diffusion model
```bash
# Extract video from .pkl file
python scripts/extract_tapvid_video.py --dataset-dir tapvid_davis --dataset davis

# Get video path for generation
python scripts/get_video_path.py --video-dir datasets

# generate real video
python scripts/generate_video.py \
    --video_file video_path.txt \
    --start_step 34 \
    --model cosmos2 \
    --num_inference_steps 35 \
    --save_steps 34 \
    --save_layers 18 \
    --save_qk \
    --resolution 704 1280 \
    --gpu 0 1 \
    --save_dir tapvid_davis_cosmos2
```

### Point Tracking
```bash
./experiments/eval_davis.sh
```
