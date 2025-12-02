#!/bin/bash

set -eu
source "configs/cosmos_config.sh"

dataset=davis
dataset_dir=tapvid_davis
feature_save_dir=tapvid_davis_cosmos2
track_save_dir=output/track
eval_save_dir=output/eval
gpu="0"

python scripts/get_trajectory.py \
    --data-dir "$feature_save_dir" \
    --dataset-dir "$dataset_dir" \
    --dataset $dataset \
    --step "$step" \
    --layer "$layer" \
    --head "$head" \
    --query-feature-type "$query_feature_type" \
    --target-feature-type "$target_feature_type" \
    --update-feature-type "$update_feature_type" \
    --upsample-feature \
    --freq-range $freq_range \
    --argmax-radius $argmax_radius \
    --search-radius $search_radius \
    --vis-threshold $vis_threshold \
    --feature-ema-alpha $feature_ema_alpha \
    --feature-update-sampling-range $feature_update_sampling_range \
    --output-dir "$track_save_dir" \
    --gpu $gpu


python scripts/evaluate.py \
    --data-dir "$track_save_dir" \
    --dataset-dir "$dataset_dir" \
    --dataset $dataset \
    --step "$step" \
    --layer "$layer" \
    --head "$head" \
    --query-feature-type "$query_feature_type" \
    --target-feature-type "$target_feature_type" \
    --output-dir "$eval_save_dir" 
