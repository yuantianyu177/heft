import numpy as np
from typing import Union, Tuple, List


def compute_occlusion_accuracy(
    pred_visibility: List[np.ndarray],
    gt_visibility: List[np.ndarray],
    video_level_average: bool = False,
) -> np.ndarray:
    """Compute occlusion accuracy metric.
    
    Args:
        pred_visibility: List of predicted visibility masks, each of shape [n, t]
        gt_visibility: List of ground truth visibility masks, each of shape [n, t]
        video_level_average: If True, compute per-video metrics and average them;
                            If False, aggregate all videos and compute single metric
        
    Returns:
        Occlusion accuracy value
    """
    if video_level_average:
        # Compute metric for each video separately and then average
        video_metrics = []
        for pred_vis, gt_vis in zip(pred_visibility, gt_visibility):
            # Compute matches for this video
            matches = np.equal(pred_vis, gt_vis)
            
            # Compute accuracy for this video
            total_points = np.sum(np.ones_like(gt_vis, dtype=np.int32))
            if total_points > 0:
                video_metric = np.sum(matches) / total_points
            else:
                video_metric = 0.0
            
            video_metrics.append(video_metric)
        
        # Average across videos
        return np.mean(video_metrics)
    else:
        # Original behavior: concatenate all videos together
        all_pred_visibility = np.concatenate(pred_visibility, axis=0)  # [total_tracks, t]
        all_gt_visibility = np.concatenate(gt_visibility, axis=0)      # [total_tracks, t]
        
        # Compute matches across all data
        matches = np.equal(all_pred_visibility, all_gt_visibility)
        # Compute overall accuracy: scalar
        total_points = np.sum(np.ones_like(all_gt_visibility, dtype=np.int32))
        occ_acc = np.sum(matches) / total_points
        
        return occ_acc


def compute_pts_within_thresh(
    pred_tracks: List[np.ndarray],
    gt_tracks: List[np.ndarray],
    visibility: List[np.ndarray],
    thresh: Union[int, float],
    video_level_average: bool = False,
) -> np.ndarray:
    """Compute fraction of points within threshold metric.
    
    Args:
        pred_tracks: List of predicted tracks, each of shape [n, t, 2]
        gt_tracks: List of ground truth tracks, each of shape [n, t, 2]  
        visibility: List of visibility masks, each of shape [n, t]
        thresh: Distance threshold in pixels
        video_level_average: If True, compute per-video metrics and average them;
                            If False, aggregate all videos and compute single metric
        
    Returns:
        Fraction of points within threshold
    """
    if video_level_average:
        # Compute metric for each video separately and then average
        video_metrics = []
        for pred, gt, vis in zip(pred_tracks, gt_tracks, visibility):
            # Compute distances and check if within threshold
            within_dist = np.sum(
                np.square(pred - gt),
                axis=-1,
            ) < np.square(thresh)
            
            # Only consider visible points in ground truth
            is_correct = np.logical_and(within_dist, vis)
            
            # Compute metrics for this video
            count_correct = np.sum(is_correct)
            count_visible_points = np.sum(vis)
            
            # Avoid division by zero
            if count_visible_points > 0:
                video_metric = count_correct / count_visible_points
            else:
                video_metric = 0.0
            
            video_metrics.append(video_metric)
        
        # Average across videos
        return np.mean(video_metrics)
    else:
        # Original behavior: concatenate all videos together
        all_pred_tracks = np.concatenate(pred_tracks, axis=0)  # [total_tracks, t, 2]
        all_gt_tracks = np.concatenate(gt_tracks, axis=0)      # [total_tracks, t, 2]
        all_visibility = np.concatenate(visibility, axis=0)     # [total_tracks, t]
        
        # Compute distances and check if within threshold
        within_dist = np.sum(
            np.square(all_pred_tracks - all_gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        
        # Only consider visible points in ground truth
        is_correct = np.logical_and(within_dist, all_visibility)
        
        # Compute overall metrics: scalar
        count_correct = np.sum(is_correct)
        count_visible_points = np.sum(all_visibility)
        
        # Avoid division by zero
        frac_correct = np.divide(
            count_correct, 
            count_visible_points,
            out=np.zeros_like(count_correct, dtype=np.float64),
            where=count_visible_points != 0
        )
        
        return frac_correct


def compute_jaccard(
    pred_tracks: List[np.ndarray],
    gt_tracks: List[np.ndarray],
    pred_visibility: List[np.ndarray],
    gt_visibility: List[np.ndarray],
    thresh: Union[int, float],
    video_level_average: bool = False,
) -> np.ndarray:
    """Compute Jaccard metric.
    
    Args:
        pred_tracks: List of predicted tracks, each of shape [n, t, 2]
        gt_tracks: List of ground truth tracks, each of shape [n, t, 2]
        pred_visibility: List of predicted visibility masks, each of shape [n, t]
        gt_visibility: List of ground truth visibility masks, each of shape [n, t]
        thresh: Distance threshold in pixels
        video_level_average: If True, compute per-video metrics and average them;
                            If False, aggregate all videos and compute single metric
        
    Returns:
        Jaccard metric value
    """
    if video_level_average:
        # Compute metric for each video separately and then average
        video_metrics = []
        for pred, gt, pred_vis, gt_vis in zip(pred_tracks, gt_tracks, pred_visibility, gt_visibility):
            # Compute distances and check if within threshold
            within_dist = np.sum(
                np.square(pred - gt),
                axis=-1,
            ) < np.square(thresh)
            
            # True positives: predicted visible, actually visible, and within threshold
            is_correct = np.logical_and(within_dist, gt_vis)
            
            # Compute metrics for this video
            true_positives = np.sum(is_correct & pred_vis)
            gt_positives = np.sum(gt_vis)
            false_positives = (~gt_vis) & pred_vis
            false_positives = false_positives | ((~within_dist) & pred_vis)
            false_positives = np.sum(false_positives)
            
            # Jaccard = TP / (TP + FP + FN) = TP / (GT_positives + FP)
            denominator = gt_positives + false_positives
            if denominator > 0:
                video_metric = true_positives / denominator
            else:
                video_metric = 0.0
            
            video_metrics.append(video_metric)
        
        # Average across videos
        return np.mean(video_metrics)
    else:
        # Original behavior: concatenate all videos together
        all_pred_tracks = np.concatenate(pred_tracks, axis=0)    # [total_tracks, t, 2]
        all_gt_tracks = np.concatenate(gt_tracks, axis=0)        # [total_tracks, t, 2]
        all_pred_visibility = np.concatenate(pred_visibility, axis=0)     # [total_tracks, t]
        all_gt_visibility = np.concatenate(gt_visibility, axis=0)         # [total_tracks, t]
        
        # Compute distances and check if within threshold
        within_dist = np.sum(
            np.square(all_pred_tracks - all_gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        
        # True positives: predicted visible, actually visible, and within threshold
        is_correct = np.logical_and(within_dist, all_gt_visibility)
        
        # Compute overall metrics: scalar
        true_positives = np.sum(is_correct & all_pred_visibility)
        gt_positives = np.sum(all_gt_visibility)
        false_positives = (~all_gt_visibility) & all_pred_visibility
        false_positives = false_positives | ((~within_dist) & all_pred_visibility)
        false_positives = np.sum(false_positives)
        
        # Jaccard = TP / (TP + FP + FN) = TP / (GT_positives + FP)
        denominator = gt_positives + false_positives
        jaccard = np.divide(
            true_positives,
            denominator,
            out=np.zeros_like(true_positives, dtype=np.float64),
            where=denominator != 0
        )
        
        return jaccard


def compute_multi_threshold_metrics(
    pred_tracks: List[np.ndarray],
    gt_tracks: List[np.ndarray],
    pred_visibility: List[np.ndarray],
    gt_visibility: List[np.ndarray],
    thresholds: list = [1, 2, 4, 8, 16],
    video_level_average: bool = False,
) -> Tuple[dict, dict]:
    """Compute metrics for multiple thresholds.
    
    Args:
        pred_tracks: List of predicted tracks, each of shape [n, t, 2]
        gt_tracks: List of ground truth tracks, each of shape [n, t, 2]
        pred_visibility: List of predicted visibility masks, each of shape [n, t]
        gt_visibility: List of ground truth visibility masks, each of shape [n, t]
        thresholds: List of distance thresholds to evaluate
        video_level_average: If True, compute per-video metrics and average them;
                            If False, aggregate all videos and compute single metric
        
    Returns:
        Tuple of (pts_within_thresh_metrics, jaccard_metrics)
    """
    pts_within_metrics = {}
    jaccard_metrics = {}
    all_pts_within = []
    all_jaccard = []
    
    for thresh in thresholds:
        # Compute pts within threshold
        pts_within = compute_pts_within_thresh(
            pred_tracks, gt_tracks, gt_visibility, thresh, video_level_average
        )
        pts_within_metrics[f'pts_within_{thresh}'] = pts_within
        all_pts_within.append(pts_within)
        
        # Compute Jaccard
        jaccard = compute_jaccard(
            pred_tracks, gt_tracks, pred_visibility, gt_visibility, 
            thresh, video_level_average
        )
        jaccard_metrics[f'jaccard_{thresh}'] = jaccard
        all_jaccard.append(jaccard)
    
    # Compute averages across thresholds (aggregated results)
    avg_pts_within = np.mean(all_pts_within, axis=0)
    avg_jaccard = np.mean(all_jaccard, axis=0)
    
    pts_within_metrics['average_pts_within_thresh'] = avg_pts_within
    jaccard_metrics['average_jaccard'] = avg_jaccard
    
    return pts_within_metrics, jaccard_metrics
