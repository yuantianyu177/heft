import os
import torch
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from multiprocessing import Process, Semaphore
from tqdm import tqdm
from itertools import product
from heft.tracker import Tracker
from heft.data.tapvid import TapVid, TapVidRGBStack, TapVidKinetics
from heft.utils.geometry import get_grid

feature_type = ["query", "key", "hidden_states"]
def parse_args():
    parser = argparse.ArgumentParser(description='Extract tracks from diffusion tracker')
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--gpu', type=int, nargs='+', default=[0])
    parser.add_argument('--dataset', type=str, default='davis',choices=['davis', 'rgb-stacking', 'kinetics'])
    parser.add_argument('--dataset-dir', type=str, default='.')
    parser.add_argument('--step', type=int, nargs='+', required=True)
    parser.add_argument('--layer', type=int, nargs='+', required=True)
    parser.add_argument('--head', type=int, nargs='+', required=True)
    parser.add_argument('--dense-query-points', action='store_true')
    parser.add_argument('--query-points-dir', type=str, default=None, help='Directory containing query points .pt files named as 04d.pt (e.g., 0000.pt, 0001.pt)')
    parser.add_argument('--query-feature-type', type=str, default='query', choices=feature_type)
    parser.add_argument('--target-feature-type', type=str, default='key', choices=feature_type)
    parser.add_argument('--update-feature-type', type=str, default=None)
    parser.add_argument('--patch-size', type=int, default=16)
    parser.add_argument('--argmax-radius', type=float, default=35.)
    parser.add_argument('--search-radius', type=float, default=100.)
    parser.add_argument('--vis-threshold', type=float, default=16.)
    parser.add_argument('--feature-ema-alpha', type=float, default=0.0)
    parser.add_argument('--feature-update-sampling-range', type=int, default=1)
    parser.add_argument('--upsample-feature', action='store_true')
    parser.add_argument('--freq-range', type=float, nargs=2, default=(0.0, 1.0))
    parser.add_argument('--resolution', type=int, nargs=2, default=(480, 832))
    parser.add_argument('--rope-dim', type=int, nargs=3, default=(44, 42, 42))
    return parser.parse_args()


def get_query_points_from_davis(dataset_dir: str, output_size: tuple = (480, 832)):
    dataset = TapVid(root=dataset_dir)
    query_points_list = dataset.get_query_points(output_size=output_size)
    query_points_list = [torch.from_numpy(query_points) for query_points in query_points_list]
    return query_points_list

def get_query_points_from_rgb_stacking(dataset_dir: str, output_size: tuple = (480, 832)):
    dataset = TapVidRGBStack(root=dataset_dir)
    query_points_list = dataset.get_query_points(output_size=output_size)
    query_points_list = [torch.from_numpy(query_points) for query_points in query_points_list]
    return query_points_list

def get_query_points_from_kinetics(dataset_dir: str, output_size: tuple = (480, 832)):
    dataset = TapVidKinetics(root=dataset_dir)
    query_points_list = dataset.get_query_points(output_size=output_size)
    query_points_list = [torch.from_numpy(query_points) for query_points in query_points_list]
    return query_points_list

def get_query_points_from_dir(query_points_dir: str) -> list:
    """Load query points from directory containing .pt files.
    
    Args:
        query_points_dir: Directory containing .pt files named as 04d.pt (e.g., 0000.pt, 0001.pt).
                         Each file should contain a tensor of shape (n, 3) where n is the number
                         of query points, and the last dimension is (x, y, t)
    
    Returns:
        List of tensors, each element is a tensor of shape (n, 3)
    """
    query_points_dir = Path(query_points_dir)
    query_points_list = []
    
    # Find all .pt files matching the 04d.pt pattern and sort them numerically
    pt_files = sorted(query_points_dir.glob("[0-9][0-9][0-9][0-9].pt"), key=lambda x: int(x.stem))
    
    if not pt_files:
        raise ValueError(f"No .pt files found in {query_points_dir} matching pattern 04d.pt")
    
    # Load each file and append to list
    for pt_file in pt_files:
        query_points = torch.load(pt_file)  # (n, 3)
        if query_points.dim() != 2 or query_points.shape[1] != 3:
            raise ValueError(f"Query points file {pt_file} should have shape (n, 3), got {query_points.shape}")
        query_points_list.append(query_points)
    
    return query_points_list


def get_dense_query_points_with_mask(task_dir: Path, patch_size: int = 64, resolution: tuple = (480, 832)) -> torch.Tensor:
    """Get dense query points, filtered by mask if available."""
    grid = get_grid(resolution[0], resolution[1], patch_size=patch_size)  # (H, W, 2)
    points = grid.reshape(-1, 2)  # (H*W, 2)
    
    # Check if mask exists
    mask_path = task_dir / "mask" / "00000.png"
    if mask_path.exists():
        mask = Image.open(mask_path) # (H, W)
        mask = mask.resize((resolution[1], resolution[0]), resample=Image.LANCZOS)
        mask = np.array(mask) # (H, W)
        mask = torch.from_numpy(mask).bool() # (H, W)
        x_coords = points[:, 0].long()
        y_coords = points[:, 1].long()
        point_mask = torch.zeros(points.shape[0], dtype=torch.bool)
        mask_values = mask[y_coords, x_coords]
        point_mask = mask_values.bool()
        points = points[point_mask]
    
    points = torch.cat([points, torch.zeros(points.shape[0], 1)], dim=-1) # (N, 3)
    return points


def process_task(task_dir, args, query_points):
    tracker = Tracker(
        data_dir=str(task_dir),
        patch_size=args.patch_size,
        argmax_radius=args.argmax_radius,
        search_radius=args.search_radius,
        vis_threshold=args.vis_threshold,
        feature_ema_alpha=args.feature_ema_alpha,
        feature_update_sampling_range=args.feature_update_sampling_range,
        freq_range=args.freq_range,
        rope_dim=args.rope_dim,
    )
    
    try:
        configs = list(product(args.step, args.layer, args.head))
        for step, layer, head in tqdm(configs, desc="Processing", ncols=80):
            if head == -1:
                tracker.layer_track(
                    step=step, layer=layer,
                    query_points=query_points,
                    query_feature_type=args.query_feature_type,
                    target_feature_type=args.target_feature_type,
                    update_feature_type=args.update_feature_type,
                    output_dir=args.output_dir + "/" + task_dir.name if args.output_dir is not None else None,
                    upsample_feature=args.upsample_feature,
                )
            else:
                tracker.head_track(
                    step=step, layer=layer, head=head,
                    query_points=query_points,
                    query_feature_type=args.query_feature_type,
                    target_feature_type=args.target_feature_type,
                    update_feature_type=args.update_feature_type,
                    output_dir=args.output_dir + "/" + task_dir.name if args.output_dir is not None else None,
                    upsample_feature=args.upsample_feature,
                )
    except Exception as e:
        print(f"Error processing {task_dir.name}, step {step}, layer {layer}, head {head}: {e}")
    finally:
        del tracker
        torch.cuda.empty_cache()


def worker(gpu_id, params, sem):
    with sem:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Process {params[0].name}, using GPU {gpu_id}")
        process_task(*params)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    task_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    # Get query points
    if args.query_points_dir is not None:
        query_points_list = get_query_points_from_dir(args.query_points_dir)
    elif args.dense_query_points:
        query_points_list = []
        for task_dir in task_dirs:
            query_points = get_dense_query_points_with_mask(task_dir, patch_size=32, resolution=args.resolution)
            query_points_list.append(query_points)
    elif args.dataset == "davis":
        query_points_list = get_query_points_from_davis(args.dataset_dir, args.resolution)
    elif args.dataset == "rgb-stacking":
        query_points_list = get_query_points_from_rgb_stacking(args.dataset_dir, args.resolution)
    elif args.dataset == "kinetics":
        query_points_list = get_query_points_from_kinetics(args.dataset_dir, args.resolution)
    else:
        raise ValueError(f"Invalid dataset: {args.dataset}")
    
    assert len(query_points_list) == len(task_dirs), f"Query points: {len(query_points_list)}, Task dirs: {len(task_dirs)}"

    gpu_sems = {gpu: Semaphore(1) for gpu in args.gpu}
    processes = []
    for i, task_dir in enumerate(task_dirs):
        gpu_id = args.gpu[i % len(args.gpu)]
        sem = gpu_sems[gpu_id]
        params = (task_dir, args, query_points_list[i])
        p = Process(target=worker, args=(gpu_id, params, sem))
        p.start()
        processes.append(p)   

    for p in processes:
        p.join()
        

if __name__ == "__main__":
    main()

