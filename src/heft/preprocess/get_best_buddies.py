import torch
from pathlib import Path
from tqdm import tqdm
from einops import rearrange
from torch import einsum
from torchvision.ops import batched_nms
from diffusion_tracker.utils.geometry import create_meshgrid


@torch.no_grad()
def get_best_buddies(
    feature_path: str, 
    height: int, 
    width: int, 
    output_path: str, 
    device:str="cuda:0"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    best_buddies = {}
    coords_grid = create_meshgrid(height, width).to(device)
    coords_grid = rearrange(coords_grid, 'h w c -> (h w) c') # (H*W, 2)
    
    features = torch.load(feature_path) # (T, C, H, W)
    features = rearrange(features, 't c h w -> t (h w) c') # (T, H*W, C)

    T = features.shape[0]
    for source_t in tqdm(range(T), desc="source time"):
        for target_t in tqdm(range(T), desc="target time"):
            if source_t == target_t:
                continue
            
            source_features = features[source_t].to(device) # (H*W, C)
            target_features = features[target_t].to(device) # (H*W, C)
            index_range = torch.arange(source_features.shape[0]).to(device) # (H*W,)

            affinity = torch.einsum("nc,mc->nm", source_features, target_features) # (H*W, H*W)
            affinity = affinity / torch.clamp(source_features.norm(dim=1)[:, None] * target_features.norm(dim=1)[None, ...], min=1e-08) # (H*W, H*W)
            affinity_source_max = torch.argmax(affinity, dim=1) # (H*W,)
            affinity_target_max = torch.argmax(affinity, dim=0) # (H*W,)
            source_bb_indices = index_range == affinity_target_max[affinity_source_max]
            target_bb_indices = affinity_source_max[source_bb_indices]

            source_coords = coords_grid[source_bb_indices]
            target_coords = coords_grid[target_bb_indices]
            affinities = affinity[index_range[source_bb_indices], target_bb_indices]
            
            best_buddies[f'{source_t}_{target_t}'] = {
                "source_coords": source_coords,
                "target_coords": target_coords,
                "cos_sims": affinities
            }
    
    torch.save(best_buddies, output_path)


def get_closest_traj_idx(trajectories, points, t, batch_size=100):
    """ 
    Args:
        trajectories: N x T x 2
        points: B x 2
        t: int
    returns: B
        """
    # sample trajectories at time t
    trajectories_at_t = trajectories[:, t, :] # N x 2
    # iterate over points in batches of size batch_size
    closest_traj_idx_list = []
    for i in range(0, len(points), batch_size):
        end = min(i + batch_size, len(points))
        points_batch = points[i:end]
        source_dist = torch.norm(trajectories_at_t[None, ...] - points_batch[:, None, :], dim=2) # B x N
        source_dist = torch.nan_to_num(source_dist, nan=torch.inf)
        # comput argmin on the last dimension
        closest_traj_idx_list.append(source_dist.argmin(dim=-1)) # B
    closest_traj_idx = torch.cat(closest_traj_idx_list, dim=0)
    return closest_traj_idx


@torch.no_grad()
def filter_best_buddies(
    best_buddies_path: str,
    traj_path: str,
    output_path: str,
    height: int,
    width: int,
    device: str = "cuda:0"
):
    best_buddies = torch.load(best_buddies_path)
    traj = torch.load(traj_path).to(device)

    _, T, _ = traj.shape
    grid = create_meshgrid(height, width).to(device) # (H, W, 2)
    feature_h, feature_w = grid.shape[:2]
    grid = rearrange(grid, 'h w c -> (h w) c') # (H*W, 2)
    closest_traj_idx_list = []
    for t in tqdm(range(T), desc="pre-computing trajectory indices"):
        closest_traj_idx = get_closest_traj_idx(traj, grid, t, 30) # B, that is, len(grid)
        closest_traj_idx = closest_traj_idx.reshape(feature_h, feature_w) # (H, W)
        closest_traj_idx_list.append(closest_traj_idx)
    closest_traj_idx = torch.stack(closest_traj_idx_list, dim=0) # (T, H, W)

    traj_is_point_invalid = traj.isnan().any(dim=-1).to(device) # (N, T)
    del traj
    torch.cuda.empty_cache()
    total_filtered_bb = {}
    
    for source_t in tqdm(range(T), desc="source frames"):
        for target_t in tqdm(range(T), desc="target frames"):
            if source_t == target_t:
                continue

            source_points = best_buddies[f'{source_t}_{target_t}']['source_coords'] # N x 2; image coordinates
            target_points = best_buddies[f'{source_t}_{target_t}']['target_coords'] # N x 2; image coordinates
            cos_sims = best_buddies[f'{source_t}_{target_t}']['cos_sims']
            peak_coords = best_buddies[f'{source_t}_{target_t}'].get('peak_coords', None)
            peak_affs = best_buddies[f'{source_t}_{target_t}'].get('peak_affs', None)
            r = best_buddies[f'{source_t}_{target_t}'].get('r', None)

            filtered_bb = {
                'source_coords': None,
                'target_coords': None,
                'cos_sims': None,
                'peak_coords' : None,
                'peak_affs' : None,
                'r' : None,
            }

            # transform source_points to feature coordinates
            source_points_grid_idx = ((source_points - 8) // 16).long() # (N, 2), in (x, y) format
            target_points_grid_idx = ((target_points - 8) // 16).long() # (N, 2), in (x, y) format
            # sample closest traj indices
            closest_traj_idx_grid_at_source_t = closest_traj_idx[source_t] # (H, W)
            closest_traj_idx_grid_at_target_t = closest_traj_idx[target_t] # (H, W)
            source_points_traj_indices = closest_traj_idx_grid_at_source_t[source_points_grid_idx[:, 1], source_points_grid_idx[:, 0]] # (N,)
            target_points_traj_indices = closest_traj_idx_grid_at_target_t[target_points_grid_idx[:, 1], target_points_grid_idx[:, 0]] # (N,)

            should_sample_bb = (traj_is_point_invalid[source_points_traj_indices, target_t] & traj_is_point_invalid[target_points_traj_indices, source_t])
            
            filtered_bb['source_coords'] = source_points[should_sample_bb] if should_sample_bb.any() else None
            filtered_bb['target_coords'] = target_points[should_sample_bb] if should_sample_bb.any() else None
            filtered_bb['cos_sims'] = cos_sims[should_sample_bb] if should_sample_bb.any() else None
            if peak_coords is not None:
                filtered_bb['peak_coords'] = peak_coords[should_sample_bb] if should_sample_bb.any() else None
            if peak_affs is not None:
                filtered_bb['peak_affs'] = peak_affs[should_sample_bb] if should_sample_bb.any() else None
            if r is not None:
                filtered_bb['r'] = r[should_sample_bb] if should_sample_bb.any() else None
            total_filtered_bb[f'{source_t}_{target_t}'] = filtered_bb
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(total_filtered_bb, output_path)


def compute_max_r(bb, bb_rev):
    for i in range(bb['target_coords'].shape[0]):
        r = bb['r'][i]
        target_coord = bb['target_coords'][i]
        rev_idx = torch.norm(bb_rev['source_coords'] - target_coord[None, :], dim=1).argmin(0)
        assert torch.norm(bb_rev['target_coords'][rev_idx] - bb['source_coords'][i]) == 0
        rev_r = bb_rev['r'][rev_idx]
        max_r = max(rev_r, r)
        bb['r'][i] = max_r
        bb_rev['r'][rev_idx] = max_r
    return bb, bb_rev


def get_bb_sim_indices(affs_batched, coords, box_size=50, iou_thresh=0.5, topk=400, device="cuda:0"):
    """  affs_batched: B x N """
    topk = torch.topk(affs_batched, k=topk, sorted=False, dim=1)
    filt_idx = topk.indices # B x topk
    affs_filt = topk.values # B x topk

    if affs_filt.shape[0] == 0:
        return None, None, None
    
    filt_coords = coords[filt_idx]
    xmin = filt_coords[:, :, 0] - box_size # B x topk
    xmax = filt_coords[:, :, 0] + box_size # B x topk
    ymin = filt_coords[:, :, 1] - box_size # B x topk
    ymax = filt_coords[:, :, 1] + box_size # B x topk
    # concat to get boxes shaped B x topk x 4
    boxes = torch.cat([xmin[:, :, None], ymin[:, :, None], xmax[:, :, None], ymax[:, :, None]], dim=-1) # B x topk x 4
    # get idxs shaped (B x topk) representing the batch index
    idxs = torch.arange(filt_idx.shape[0], device=device)[:, None].repeat(1, filt_idx.shape[1]).reshape(-1) # (B x topk)
    peak_indices = batched_nms(boxes.reshape(-1, 4), affs_filt.reshape(-1), idxs, iou_thresh)
    # convert peak_indices to the original indices to the  not flat indices
    peak_indices_original = torch.stack([peak_indices // filt_idx.shape[1], peak_indices % filt_idx.shape[1]], dim=-1)
    # retrieve the first two elements of the peak_indices_original for the first axis
    filt_idx_mask = torch.zeros_like(filt_idx, device=device) # B x topk
    filt_idx_mask[peak_indices_original[:, 0], peak_indices_original[:, 1]] = 1
    peak_aff_batched = affs_filt * filt_idx_mask # B x topk
    # retrieve the highest and second highest affinities for each batch
    top2 = torch.topk(peak_aff_batched, k=2, dim=1)
    top2_values, top2_indices = top2.values, top2.indices # B x 2, B x 2
    highest_affs, _ = top2_values[:, 0], top2_indices[:, 0] # B, B
    second_highest_affs, _ = top2_values[:, 1], top2_indices[:, 1] # B, B
    r = second_highest_affs / highest_affs 
    return None, top2_values, r


def compute_bb_nms(best_buddies_sf_tf, sf, tf, features, coords, box_size, iou_thresh):
    source_xy = best_buddies_sf_tf['source_coords']
    source_fxy = ((source_xy - 8) / 16) # (N, 2)

    target_fmap = features[tf] # C x H x W
    source_f = features[sf, :, source_fxy[:, 1].int(), source_fxy[:, 0].int()] # C x N
    
    source_target_sim = einsum('cn,chw->nhw', source_f, target_fmap)
    source_f_norm = torch.norm(source_f, dim=0) # N
    target_fmap_norm = torch.norm(target_fmap, dim=0) # H x W
    source_target_sim /= torch.clamp(source_f_norm[:, None, None] * target_fmap_norm[None, :, :], min=1e-08) # N x H x W
    N = source_target_sim.shape[0]

    source_target_sim_flat = source_target_sim.reshape(N, -1) # N x (HxW)
    
    _, peak_aff, r = get_bb_sim_indices(source_target_sim_flat, coords, box_size=box_size, iou_thresh=iou_thresh)
    best_buddies_sf_tf['peak_coords'] = None
    best_buddies_sf_tf['peak_affs'] = peak_aff # N x 2
    best_buddies_sf_tf['r'] = r # N

    return best_buddies_sf_tf

def compute_nms(
    best_buddies_path: str,
    feature_path: str,
    output_path: str,
    height: int,
    width: int,
    iou_thresh: float = 0.5,
    box_size: int = 16,
    device: str = "cuda:0"
):
    best_buddies = torch.load(best_buddies_path)
    features = torch.load(feature_path).to(device) # t x c x h x w
    coords = create_meshgrid(height, width).to(device) # (H, W, 2)
    coords = rearrange(coords, 'h w c -> (h w) c') # (H*W, 2)

    for key in tqdm(best_buddies.keys()):
        if best_buddies[key]['source_coords'] is None:
            best_buddies[key]['peak_coords'] = None
            best_buddies[key]['peak_affs'] = None
            best_buddies[key]['r'] = None
            continue
        
        if best_buddies[key].get('r', None) is not None:
            continue

        sf, tf = int(key.split("_")[0]), int(key.split("_")[1])
        
        bb = compute_bb_nms(best_buddies[f'{sf}_{tf}'], sf, tf, features, coords, box_size, iou_thresh)
        bb_rev = compute_bb_nms(best_buddies[f'{tf}_{sf}'], tf, sf, features, coords, box_size, iou_thresh)
        
        bb, bb_rev = compute_max_r(bb, bb_rev)
        
        best_buddies[key] = bb
        best_buddies[f'{tf}_{sf}'] = bb_rev

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best_buddies, output_path)

