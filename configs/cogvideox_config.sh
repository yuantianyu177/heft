# ===== config.sh =====
query_feature_type=key
target_feature_type=key
update_feature_type=key
argmax_radius=13
search_radius=43
vis_threshold=27
freq_range="0.0 1.00"
feature_ema_alpha=0.1
feature_update_sampling_range=1
step=49
layer=15
head=19

dataset=davis
dataset_dir=datasets/tapvid_davis
feature_save_dir=/share/yty/tapvid_davis_cogvideox
resolution="480 720"
rope_dim="16 24 24"