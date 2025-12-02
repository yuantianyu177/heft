import torch
from torch import Tensor
import argparse
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from itertools import product
from tqdm import tqdm

from heft.utils.metric import (
    compute_occlusion_accuracy,
    compute_multi_threshold_metrics,
)
from heft.data.tapvid import TapVid, TapVidRGBStack, TapVidKinetics
from heft.utils.misc import convert_np

feature_type = ["query", "key", "hidden_states"]
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate tracking results')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='davis', choices=['davis', 'rgb-stacking', 'kinetics'])
    parser.add_argument('--dataset-dir', type=str, default='.')
    parser.add_argument('--dense-query-points', action='store_true')
    parser.add_argument('--step', type=int, nargs='+', required=True)
    parser.add_argument('--layer', type=int, nargs='+', required=True)
    parser.add_argument('--head', type=int, nargs='+', required=True)
    parser.add_argument('--query-feature-type', type=str, default='query', choices=feature_type)
    parser.add_argument('--target-feature-type', type=str, default='key', choices=feature_type)
    parser.add_argument('--output-dir', type=str, default='eval_results')
    parser.add_argument('--thresholds', type=int, nargs='+', default=[1, 2, 4, 8, 16])
    parser.add_argument('--resolution', type=int, nargs=2, default=[480, 832])
    return parser.parse_args()


def load_pseudo_tracks(task_dirs: List[Path]) -> Tuple[List[Tensor], List[Tensor]]:
    pseudo_tracks_list = []
    pseudo_visibility_list = []
    for task_dir in task_dirs:
        pseudo_tracks_path = task_dir / "pseudo_trajectory" / "tracks.pt"
        pseudo_visibility_path = task_dir / "pseudo_trajectory" / "visibility.pt"
        pseudo_tracks = torch.load(pseudo_tracks_path, map_location='cpu') # (N, T, 2)
        pseudo_visibility = torch.load(pseudo_visibility_path, map_location='cpu') # (N, T)
        pseudo_tracks_list.append(pseudo_tracks)
        pseudo_visibility_list.append(pseudo_visibility)
    return pseudo_tracks_list, pseudo_visibility_list


def load_predicted_tracks(
    task_dirs: List[Path], 
    step: int, layer: int, head: int,
    query_feature_type: str,
    target_feature_type: str,
    resolution: Tuple[int, int] = (480, 832),
    output_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Load predicted tracks and visibility from saved files.
    
    Args:
        task_dir: Directory containing the task data
        step: Diffusion step
        layer: Layer index
        head: Head index  
        query_feature_type: Type of query features
        target_feature_type: Type of target features
        output_size: Target size (height, width) to resize tracks to
    
    Returns:
        Tuple of (tracks, visibility) as numpy arrays with tracks resized to output_size
    """
    tracks_list = []
    visibility_list = []
    for task_dir in task_dirs:
        if head != -1:
            sub_dir = f"head_track/{query_feature_type}-{target_feature_type}"
            track_path = task_dir / sub_dir / f"step_{step:03d}" / f"layer_{layer:02d}" / f"head_{head:02d}_track.pt"
            visibility_path = task_dir / sub_dir / f"step_{step:03d}" / f"layer_{layer:02d}" / f"head_{head:02d}_visibility.pt"
        else:
            sub_dir = f"layer_track/{query_feature_type}-{target_feature_type}"
            track_path = task_dir / sub_dir / f"step_{step:03d}" / f"layer_{layer:02d}_track.pt"
            visibility_path = task_dir / sub_dir / f"step_{step:03d}" / f"layer_{layer:02d}_visibility.pt"
        tracks = torch.load(track_path, map_location='cpu').numpy()  # (B, T, 2)
        tracks[:, :, 0] = tracks[:, :, 0] / resolution[1] * (output_size[1] - 1)
        tracks[:, :, 1] = tracks[:, :, 1] / resolution[0] * (output_size[0] - 1)
        visibility = torch.load(visibility_path, map_location='cpu').numpy()  # (B, T)
        tracks_list.append(tracks)
        visibility_list.append(visibility)
    return tracks_list, visibility_list


def evaluate(
    gt_tracks_list: List[np.ndarray],
    gt_visibility_list: List[np.ndarray],
    pred_tracks_list: List[np.ndarray],
    pred_visibility_list: List[np.ndarray],
    thresholds: List[int],
    video_level_average: bool = True,
) -> Dict:
    """Evaluate a single configuration (step, layer, head).
    
    Args:
        gt_tracks_list: List of ground truth tracks for each video, each of shape [n, t, 2]
        gt_visibility_list: List of ground truth visibility masks for each video, each of shape [n, t]
        pred_tracks_list: List of predicted tracks for each video, each of shape [n, t, 2]
        pred_visibility_list: List of predicted visibility masks for each video, each of shape [n, t]
        thresholds: List of distance thresholds for evaluation
        video_level_average: If True, compute per-video metrics and average them;
                            If False, aggregate all videos and compute single metric
    
    Returns:
        Dictionary containing all computed metrics
    """
    cropped_gt_tracks_list = []
    cropped_gt_visibility_list = []
    cropped_pred_tracks_list = []
    cropped_pred_visibility_list = []
    
    for i, (gt_tracks, gt_vis, pred_tracks, pred_vis) in enumerate(
        zip(gt_tracks_list, gt_visibility_list, pred_tracks_list, pred_visibility_list)
    ):
        min_frames = min(gt_tracks.shape[1], pred_tracks.shape[1])
        cropped_pred_tracks_list.append(pred_tracks[:, :min_frames, :])
        cropped_pred_visibility_list.append(pred_vis[:, :min_frames])
        cropped_gt_tracks_list.append(gt_tracks[:, :min_frames, :])
        cropped_gt_visibility_list.append(gt_vis[:, :min_frames])

    results = {}
    occ_acc = compute_occlusion_accuracy(
        cropped_pred_visibility_list, cropped_gt_visibility_list, video_level_average
    )
    results['occlusion_accuracy'] = occ_acc
    
    pts_within_metrics, jaccard_metrics = compute_multi_threshold_metrics(
        cropped_pred_tracks_list, cropped_gt_tracks_list, cropped_pred_visibility_list, cropped_gt_visibility_list,
        thresholds, video_level_average
    )
    results.update(pts_within_metrics)
    results.update(jaccard_metrics)
    return results


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    if args.dense_query_points:
        gt_tracks_list, gt_visibility_list = load_pseudo_tracks(task_dirs)
    elif args.dataset == "davis":
        tapvid = TapVid(root=args.dataset_dir)
        gt_tracks_list, gt_visibility_list = tapvid.get_tracks(output_size=(256, 256))
    elif args.dataset == "rgb-stacking":
        tapvid = TapVidRGBStack(root=args.dataset_dir)
        gt_tracks_list, gt_visibility_list = tapvid.get_tracks(output_size=(256, 256))
    elif args.dataset == "kinetics":
        tapvid = TapVidKinetics(root=args.dataset_dir)
        gt_tracks_list, gt_visibility_list = tapvid.get_tracks(output_size=(256, 256))
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
   
    assert len(gt_tracks_list) == len(task_dirs), f"GT: {len(gt_tracks_list)}, Task: {len(task_dirs)}"
    
    all_results = {}
    configs = list(product(args.step, args.layer, args.head))
    for step, layer, head in tqdm(configs, desc="Evaluating", ncols=80):
        config_name = f"step_{step}_layer_{layer}_head_{head}"
        pred_tracks_list, pred_visibility_list = load_predicted_tracks(
            task_dirs, step, layer, head,
            args.query_feature_type, args.target_feature_type,
            args.resolution,
            output_size=(256, 256)
        )
        results = evaluate(
            gt_tracks_list, gt_visibility_list,
            pred_tracks_list, pred_visibility_list,
            args.thresholds,
        )
        all_results[config_name] = results
    
    sorted_results = dict(sorted(
        all_results.items(), 
        key=lambda x: x[1]['average_pts_within_thresh'], 
        reverse=True
    ))
       
    output_file = output_dir / f"evaluation_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(convert_np(sorted_results), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()