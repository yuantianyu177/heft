#!/bin/bash

set -eu
source "configs/cosmos_config.sh"

feature_save_dir=output/tapvid_davis_cosmos2
track_save_dir=output/track
eval_save_dir=output/eval
gpu="0 1 2 3 4 5 6 7"

python scripts/get_trajectory.py \
    --data-dir "$feature_save_dir" \
    --dense-query-points \
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
    --resolution $resolution \
    --rope-dim $rope_dim \
    --output-dir "$track_save_dir" \
    --gpu $gpu

