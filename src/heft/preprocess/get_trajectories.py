from pathlib import Path
import torch
from torch.nn import functional as F
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from einops import rearrange, repeat
from tqdm import tqdm
from tqdm.contrib import tzip
from ..utils.io import load_video, load_masks
from ..utils.geometry import get_grid, bilinear_sampler, bilinear_interpolate_video


@torch.no_grad()
def get_flows_with_masks(
    model,
    transforms,
    video: torch.Tensor,
    device: str = "cuda:0",
    threshold: float = 1,
):
    T, _, H, W = video.shape
    flows = torch.zeros((2, T - 1, 2, H, W), device=device)
    cycle_consistency_errors = torch.zeros((T - 1, H, W), device=device)
    valid_forward_warp = torch.zeros((T - 1, H, W), device=device, dtype=torch.bool)

    raft_video, _ = transforms(video, video)
    coords = get_grid(H, W).permute(2, 1, 0).unsqueeze(0).to(device) # (1, 2, H, W)

    for idx, (image1, image2) in enumerate(tzip(raft_video[:-1], raft_video[1:], desc="Calculating flows")):
        # Calculate flows
        image1_batch = torch.stack((image1, image2), dim=0) # shape
        image2_batch = image1_batch.flip(0)
        flow = model(image1_batch, image2_batch, num_flow_updates=24)[-1] # (2, 2, H, W)
        flow12, flow21 = flow[0:1], flow[1:2]
        flows[:, idx] = flow

        # Calculate cycle consistency errors
        coords1 = coords + flow12
        coords2 = coords1 + bilinear_sampler(flow21, coords1.permute(0, 2, 3, 1))
        cycle_consistency_errors[idx] = (coords - coords2).norm(dim=1)

        # Calculate missing forward warp
        warped_grid = coords + flow12
        warped_grid = warped_grid.round().long()
        x_valid = (warped_grid[:, 0] >= 0) & (warped_grid[:, 0] < W)
        y_valid = (warped_grid[:, 1] >= 0) & (warped_grid[:, 1] < H)
        valid_warp = x_valid & y_valid  # (B, H, W)
        valid_forward_warp[idx] = valid_warp.squeeze(0)


    masks = cycle_consistency_errors < threshold
    masks = masks & valid_forward_warp

    return flows, masks


@torch.no_grad()
def compute_direct_flows_for_start_frame(
    model,
    trasnforms,
    video: torch.Tensor,
    device: str = "cuda:0",
    threshold: float = 1,
    starting_frame : int = 0,
):  
    raft_video, _ = trasnforms(video[starting_frame:], video[starting_frame:])
    T, _, H, W = raft_video.shape
    coords = get_grid(H, W).permute(2, 1, 0).unsqueeze(0).to(device) # (1, 2, H, W)

    src_frame_batch = raft_video[0].unsqueeze(0).repeat(T-1, 1, 1, 1) # (T-1, 3, H, W)
    dst_frame_batch = raft_video[1:] # (T-1, 3, H, W)
    fwd_flows = []
    bwd_flows = []
    max_batch_size = 16
    for i in range(0, T-1, max_batch_size):
        end = min(i + max_batch_size, T)
        fwd_flows.append(model(src_frame_batch[i:end], dst_frame_batch[i:end], num_flow_updates=24)[-1])
        bwd_flows.append(model(dst_frame_batch[i:end], src_frame_batch[i:end], num_flow_updates=24)[-1])
    fwd_flows = torch.cat(fwd_flows, dim=0) # (T-1, 2, H, W)
    bwd_flows = torch.cat(bwd_flows, dim=0) # (T-1, 2, H, W)
    
    coords1 = coords + fwd_flows # (T-1, 2, H, W)
    
    # convert (x,y) coordinates to (x,y,t) coordinates to indicate frame idx
    time_grid = torch.arange(T-1).to(device) # (T-1)
    time_grid = time_grid.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # (T-1, 1, H, W)
    coords1_3d = torch.cat((coords1, time_grid), dim=1) # (T-1, 3, H, W)
    coords1_backward = bilinear_interpolate_video(
        rearrange(bwd_flows, "t c h w -> 1 c t h w"), 
        rearrange(coords1_3d, "t c h w -> (t h w) c"), h=H, w=W, t=T-1, normalize_h=True, normalize_w=True, normalize_t=True
    ) # (T-1, 2, H, W)
    coords1_backward = rearrange(coords1_backward, "(t h w) c -> t c h w", t=T-1, c=2, h=H, w=W)
    coords2 = coords1 + coords1_backward
    cycle_consistency_errors = (coords - coords2).norm(dim=1) # (T-1, H, W)
    x_valid = (coords1[:, 0] >= 0) & (coords1[:, 0] < W)
    y_valid = (coords1[:, 1] >= 0) & (coords1[:, 1] < H)
    valid_warped_grid = x_valid & y_valid
    mask = (cycle_consistency_errors < threshold) & valid_warped_grid

    return fwd_flows, mask


@torch.no_grad()
def save_trajectories(
    video_path: str, 
    output_path: str,
    resize: tuple[int, int] = None,
    threshold: float = 1, 
    min_trajectory_length: int = 2, 
    direct_flow_threshold: float = 2.5,
    filter_using_direct_flow: bool = False,
    device: str = "cuda:0"
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    video = load_video(video_path, normalize=True, resize=resize).to(device)
    T, _, H, W = video.shape

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device).eval()
    transforms = Raft_Large_Weights.DEFAULT.transforms()

    flows, valid_masks = get_flows_with_masks(
        model,
        transforms,
        video,
        device=device,
        threshold=threshold,
    ) # (2, T-1, 2, H, W), (T-1, H, W)

    fwd_flow, bwd_flow = flows[0], flows[1]

    upper_bound = torch.tensor([W, H], device=device) - 1
    all_filtered_trajectories = torch.full((0, T, 2), device="cpu", fill_value=float("nan")) # (N, T, 2)

    for starting_frame in tqdm(range(T - (min_trajectory_length - 1)), leave=False):
        trajectories = torch.zeros((T - starting_frame, H, W, 2)).float().to(device)
        coords = get_grid(H, W).permute(2, 1, 0).unsqueeze(0).to(device) # (1, 2, H, W)
        orig_coords = get_grid(H, W).permute(2, 1, 0).unsqueeze(0).to(device) # (1, 2, H, W)

        valid_mask = valid_masks[starting_frame]

        past_traj_passed = all_filtered_trajectories[:, starting_frame].to(device)
        valid_traj = past_traj_passed.isnan().any(dim=-1).logical_not() & \
            ((past_traj_passed >= 0) & (past_traj_passed <= upper_bound)).all(dim=-1)
        past_traj_passed = past_traj_passed[valid_traj] # filter out invalid trajectories
        past_traj_passed = past_traj_passed.round().long() # (M, 2)

        passed_through = torch.zeros_like(valid_mask) # if a pixel is already passed through, it cannot be used again
        passed_through[past_traj_passed[:, 1], past_traj_passed[:, 0]] = True
        valid_mask = valid_mask & ~passed_through # (H, W)

        trajectories[0] = torch.where(valid_mask.unsqueeze(-1), coords.squeeze(0).permute(1, 2, 0), float("nan"))
        
        if filter_using_direct_flow:
            dflows, dflow_masks = compute_direct_flows_for_start_frame(
                model=model,
                trasnforms=transforms,
                video=video,
                device=device,
                threshold=threshold,
                starting_frame=starting_frame,
            )

        for idx in tqdm(range(T - 1 - starting_frame), leave=False):
            if filter_using_direct_flow:
                dflow, dflow_mask = dflows[idx], dflow_masks[idx]
                dflow_coords = orig_coords + dflow.unsqueeze(0) # (1, 2, H, W)

            flow12 = fwd_flow[starting_frame + idx]
            flow21 = bwd_flow[starting_frame + idx]

            flow12_warped = bilinear_sampler(flow12.unsqueeze(0), coords.permute(0, 2, 3, 1)) # (1, 2, H, W)
            coords1 = coords + flow12_warped
            coords2 = coords1 + bilinear_sampler(flow21.unsqueeze(0), coords1.permute(0, 2, 3, 1))
            err = (coords - coords2).norm(dim=1)
            x_valid = (coords1[:, 0] >= 0) & (coords1[:, 0] < W)
            y_valid = (coords1[:, 1] >= 0) & (coords1[:, 1] < H)
            valid = x_valid & y_valid  # (B, H, W)
            valid_mask = valid_mask & (err < threshold).squeeze(0) & valid.squeeze(0)

            coords += flow12_warped
            
            if filter_using_direct_flow:
                err_dflow = (coords - dflow_coords).norm(dim=1)
                err_dflow = err_dflow * (dflow_mask.float() > 0.2).float()
                valid_mask = valid_mask & (err_dflow.squeeze(0) <  direct_flow_threshold)
            
            trajectories[idx + 1] = torch.where(valid_mask.unsqueeze(-1), coords.squeeze(0).permute(1, 2, 0), float("nan"))

        padded_trajectories = F.pad(
            rearrange(trajectories, "t h w c -> h w c t"), (starting_frame, 0), mode="constant", value=float("nan")
        )
        padded_trajectories = rearrange(padded_trajectories, "h w c t -> (h w) t c")
        one_nan_least = padded_trajectories.isnan().any(dim=-1)
        set_nans = repeat(one_nan_least, "n t -> n t 2")
        padded_trajectories[set_nans] = float("nan")
        current_not_nan_traj = padded_trajectories[padded_trajectories.isnan().any(dim=-1).logical_not().sum(dim=-1) >= min_trajectory_length]
        all_filtered_trajectories = torch.cat([all_filtered_trajectories, current_not_nan_traj.cpu()], dim=0) # (N, t, 2), (M, t, 2) -> ((M+N), t, 2)

    can_sample = all_filtered_trajectories.isnan().any(dim=-1).logical_not() # (N, T)
    valid_trajs_idx = (can_sample.sum(dim=1) > 1) # N
    # Remove trajectories that are only one frame long
    valid_trajectories = all_filtered_trajectories[valid_trajs_idx] # (N', T, 2)
    torch.save(valid_trajectories, output_path)
    print(f"Saved {output_path}, shape: {valid_trajectories.shape}")


@torch.no_grad()
def save_mask_trajectories(
    trajectory_path: str,
    mask_path: str,
    output_path: str,
    resize: tuple[int, int] = None,
    save_foreground: bool = True,
    device: str = "cuda:0"
):
    """
    Filter and save trajectories based on whether their starting point is in foreground or background.
    
    Args:
        trajectory_path: Path to trajectory .pt file with shape (N, T, 2)
        mask_path: Path to mask video file 
        output_path: Path to save filtered trajectories
        save_foreground: If True, save foreground trajectories; if False, save background trajectories
        device: Device to use for computation
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load trajectories (N, T, 2)
    trajectories = torch.load(trajectory_path, map_location="cpu")
    N, T, _ = trajectories.shape
    
    # Load mask video (T, C, H, W) and convert to binary mask
    mask_video = torch.from_numpy(load_masks(mask_path, h_resize=resize[0], w_resize=resize[1])).to(device)
    T_mask, H, W = mask_video.shape
    
    # Ensure trajectory time dimension matches mask time dimension
    if T != T_mask:
        print(f"Warning: Trajectory time dimension {T} doesn't match mask time dimension {T_mask}")
        T = min(T, T_mask)
        trajectories = trajectories[:, :T, :]
        mask_video = mask_video[:T]
    
    # Find valid trajectories based on starting points using batch processing
    valid_indices = []
    batch_size = 100000
    
    for batch_start in tqdm(range(0, N, batch_size), desc="Processing trajectory batches"):
        batch_end = min(batch_start + batch_size, N)
        batch_trajectories = trajectories[batch_start:batch_end].to(device)  # (batch_size, T, 2)
        batch_size_actual = batch_trajectories.shape[0]
        
        # Find first non-nan position for each trajectory in batch
        valid_masks = ~torch.isnan(batch_trajectories).any(dim=2)  # (batch_size, T)
        
        # Get first valid indices for each trajectory (vectorized)
        first_valid_indices = torch.argmax(valid_masks.float(), dim=1)  # (batch_size,)
        
        # Get starting positions for all trajectories
        batch_indices = torch.arange(batch_size_actual, device=device)
        start_positions = batch_trajectories[batch_indices, first_valid_indices]  # (batch_size, 2)
        
        # Convert to integer coordinates (no need for bounds checking)
        x_int = torch.round(start_positions[:, 0]).long()
        y_int = torch.round(start_positions[:, 1]).long()
        
        # Check if starting points are in foreground
        time_indices = first_valid_indices  # (batch_size,)
        is_foreground = mask_video[time_indices, y_int, x_int]  # (batch_size,)
        
        # Filter based on save_foreground flag
        target_mask = is_foreground if save_foreground else ~is_foreground
        
        # Add valid indices to the list
        batch_valid_indices = torch.where(target_mask)[0] + batch_start
        valid_indices.extend(batch_valid_indices.cpu().tolist())
    
    if len(valid_indices) == 0:
        print(f"Warning: No valid {'foreground' if save_foreground else 'background'} trajectories found")
        filtered_trajectories = torch.empty((0, T, 2))
    else:
        filtered_trajectories = trajectories[valid_indices]
    
    # Save filtered trajectories
    torch.save(filtered_trajectories, output_path)
    trajectory_type = "foreground" if save_foreground else "background"
    print(f"Saved {trajectory_type} trajectories to {output_path}, shape: {filtered_trajectories.shape}")