import torch

from einops import rearrange
from torch import Tensor
from pathlib import Path
from typing import Literal, Optional, List
from torch.nn import functional as F

from .utils.io import load_video
from .utils.geometry import sampler3d, get_grid



class Tracker:
    def __init__(
        self,
        data_dir: str,
        patch_size: int = 16,
        argmax_radius: float = 35.,
        search_radius: float = 100.,
        vis_threshold: float = 16.,
        feature_ema_alpha: float = 0.05,
        feature_update_sampling_range: int = 1,
        freq_range: tuple[float, float] = (0.0, 1.0),
        rope_dim: tuple[int, int, int] = (44, 42, 42), # (T, H, W)
        device: str = "cuda:0",
    ):
        self.data_dir = Path(data_dir)
        self.patch_size = patch_size
        self.argmax_radius = argmax_radius
        self.search_radius = search_radius
        self.vis_threshold = vis_threshold
        self.feature_ema_alpha = feature_ema_alpha
        self.feature_update_sampling_range = feature_update_sampling_range
        self.freq_range = freq_range
        self.device = device
        self.rope_dim = rope_dim
        self.set_path()
        self.set_resolution()
        
    def set_resolution(self):
        self.video = load_video(str(self.video_dir / "original_video.mp4"))
        self.frames, _, self.video_h, self.video_w = self.video.shape
        self.feature_h = self.video_h // self.patch_size
        self.feature_w = self.video_w // self.patch_size
    
    def get_num_chunk(self):
        """Get number of chunks and chunk size by checking existing files
        
        Returns:
            (num_chunks, chunk_size): Number of chunks and frames per chunk
        """
        # Find any step/layer directory to check chunk structure
        for step_dir in self.query_dir.iterdir():
            if step_dir.is_dir() and step_dir.name.startswith("step_"):
                # Count chunk files
                chunk_files = list(step_dir.glob("layer_*_chunk_*.pt"))
                if not chunk_files:
                    # No chunks found, assume single chunk with all frames
                    return 1, self.frames
                
                # Extract chunk IDs
                chunk_ids = []
                for f in chunk_files:
                    parts = f.stem.split("_chunk_")
                    if len(parts) == 2:
                        chunk_ids.append(int(parts[1]))
                
                if not chunk_ids:
                    return 1, self.frames
                
                num_chunks = max(chunk_ids) + 1
                
                # Load first two chunks to determine chunk_size
                try:
                    layer_file = chunk_files[0].name.split("_chunk_")[0]
                    chunk_0 = torch.load(step_dir / f"{layer_file}_chunk_000.pt")
                    # Shape is (1, head, seq_len, c)
                    chunk_0_frames = chunk_0.shape[2] // (self.feature_h * self.feature_w)
                    chunk_size = chunk_0_frames
                    return num_chunks, chunk_size
                except:
                    pass
        
        # Fallback to single chunk
        return 1, self.frames

    def set_path(self):
        self.head_track_dir = self.data_dir / "head_track"
        self.layer_track_dir = self.data_dir / "layer_track"
        self.query_dir = self.data_dir / "query"
        self.key_dir = self.data_dir / "key"
        self.hidden_states_dir = self.data_dir / "hidden_states"
        self.video_dir = self.data_dir / "video"
        self.feature_dir = {
            "query": self.query_dir,
            "key": self.key_dir,
            "hidden_states": self.hidden_states_dir
        }
        self.head_track_dir.mkdir(exist_ok=True)
        self.layer_track_dir.mkdir(exist_ok=True)

    def load_feature(self, type: Literal["query", "key", "hidden_states"], step: int, layer: int, head: int, chunk_id: int = None) -> Tensor:
        """Load feature, optionally concatenating all chunks
        
        Args:
            type: Feature type
            step: Step number
            layer: Layer number
            head: Head number (-1 for all heads)
            chunk_id: Specific chunk ID to load, or None to load and concatenate all chunks
            
        Returns:
            Feature tensor (T, C, H, W) or (T, head*C, H, W) if head=-1
        """
        dir = self.feature_dir[type]
        step_dir = dir / f"step_{step:03d}"
        
        if chunk_id is None:
            # Load and concatenate all chunks
            chunk_id = 0
            features = []
            while True:
                file_path = step_dir / f"layer_{layer:02d}_chunk_{chunk_id:03d}.pt"
                if not file_path.exists():
                    break
                
                feature = torch.load(file_path).to(self.device)
                features.append(rearrange(feature, "1 head (t h w) c -> head t c h w", h=self.feature_h, w=self.feature_w))
                chunk_id += 1
            
            feature = torch.cat(features, dim=1)
        else:
            file_path = step_dir / f"layer_{layer:02d}_chunk_{chunk_id:03d}.pt"
            feature = torch.load(file_path).to(self.device)
            feature = rearrange(feature, "1 head (t h w) c -> head t c h w", h=self.feature_h, w=self.feature_w)
        
        if head == -1:
            feature = rearrange(feature, "head t c h w -> t (head c) h w")
        else:
            T, H, W = self.rope_dim
            feature = feature[head]
            feature_t = feature[:, int(T * self.freq_range[0]):int(T * self.freq_range[1])]
            feature_h = feature[:, T+int(H * self.freq_range[0]):T+int(H * self.freq_range[1])]
            feature_w = feature[:, T+H+int(W * self.freq_range[0]):T+H+int(W * self.freq_range[1])]
            feature = torch.cat([feature_t, feature_h, feature_w], dim=1)
        return feature

    def upsample_feature(self, feature: Tensor, frames: int) -> Tensor:        
        feature = rearrange(feature, "t c h w -> 1 c t h w")
        feature = F.interpolate(feature, size=(frames, self.video_h, self.video_w), mode="trilinear")
        feature = rearrange(feature, "1 c t h w -> t c h w")
        return feature
    
    def interpolate_feature(self, feature: Tensor, t: int) -> Tensor:        
        feature = rearrange(feature, "t c h w -> 1 c t h w")
        feature = F.interpolate(feature, size=(t, feature.shape[3], feature.shape[4]), mode="trilinear")
        feature = rearrange(feature, "1 c t h w -> t c h w")
        return feature

    def sample_feature(self, feature: Tensor, points: Tensor, frames: int) -> Tensor:
        """
        feature: (T, C, H, W)
        points: (B, 3)
        """
        return sampler3d(feature, points, self.video_h, self.video_w, frames)

    def get_corr_maps(self, query: Tensor, target: Tensor) -> Tensor:
        """
        query: (B, C)
        target: (C, H, W)
        """
        corr_maps = torch.einsum("b c, c h w -> b h w", query, target)
        query_norm = query.norm(dim=1) # (B,)
        target_norm = target.norm(dim=0) # (H, W)
        query_norm = query_norm.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
        target_norm = target_norm.unsqueeze(0) # (1, H, W)
        norm = (query_norm * target_norm) # (B, H, W)
        corr_maps = corr_maps / torch.clamp(norm, min=1e-08)      
        return corr_maps # (B, H, W)

    def softmax_corr_map(self, corr_maps: Tensor) -> Tensor:
        """
        corr_maps: (B, H, W)
        """
        _, h, w = corr_maps.shape
        corr_maps = rearrange(corr_maps, "b h w -> b (h w)")
        corr_maps = F.softmax(corr_maps, dim=-1)
        corr_maps = rearrange(corr_maps, "b (h w) -> b h w", h=h)
        return corr_maps

    def ema_update_query_feature(
        self, 
        updated_query_feature: Tensor, 
        query_feature_map: Tensor, 
        positions: Tensor, 
        visible_mask: Tensor, 
        time_step: int,
    ) -> Tensor:
        """
        Update query features using EMA with weighted neighborhood features.
        
        Args:
            updated_query_feature: Current query features (B, C)
            query_feature_map: Query feature map for current frame (T, C, H, W)
            positions: Current positions (B, 2)
            visible_mask: Current visible mask (B,)
            time_step: Current time step
        Returns:
            updated_query_feature: Updated query features (B, C)
        """ 
        # Get sample points
        num_visible = visible_mask.sum()
        if num_visible == 0:
            return updated_query_feature
        tracked_centers = positions[visible_mask]  # (N_visible, 2)
        offset_x = torch.arange(-self.feature_update_sampling_range, self.feature_update_sampling_range+1, 1)
        offset_y = torch.arange(-self.feature_update_sampling_range, self.feature_update_sampling_range+1, 1)
        offsets = torch.stack(torch.meshgrid(offset_x, offset_y, indexing='xy'), dim=-1)
        offsets = offsets.reshape(-1, 2).to(self.device).to(torch.float32) # (N_offsets, 2)
        num_offsets = offsets.shape[0]
        sample_points = tracked_centers.unsqueeze(1) + offsets.unsqueeze(0) # (N_visible, N_offsets, 2)
        sample_points[..., 0] = torch.clamp(sample_points[..., 0], min=0, max=self.video_w - 1)
        sample_points[..., 1] = torch.clamp(sample_points[..., 1], min=0, max=self.video_h - 1)
        sample_points = torch.cat([sample_points, torch.full((num_visible, num_offsets, 1), time_step, device=self.device)], dim=-1)
        weights = torch.exp(-torch.norm(offsets, dim=-1) / (self.feature_update_sampling_range / 3.0)) # (N_offsets,)
        weights = weights / weights.sum(dim=-1, keepdim=True) # (N_offsets,)
        weights = weights.repeat(num_visible, 1) # (N_visible, N_offsets)
        
        # Sample query features
        points = rearrange(sample_points, "n_v n_o c -> (n_v n_o) c")
        sampled_features = self.sample_feature(query_feature_map, points, query_feature_map.shape[0])  # (N_visible * N_offsets, C)
        sampled_features = sampled_features.view(num_visible, num_offsets, -1)  # (N_visible, N_offsets, C)
        
        # Compute weighted average features
        weighted_features = torch.sum(sampled_features * weights.unsqueeze(-1), dim=1)  # (N_visible, C)
        
        # Update query features
        updated_query_feature[visible_mask] = (
            self.feature_ema_alpha * weighted_features + \
            (1 - self.feature_ema_alpha) * updated_query_feature[visible_mask]
        )
        return updated_query_feature

    def get_visibility(
        self, 
        query_points: Tensor, 
        target_points: Tensor, 
        time_step: int,
        query_feature_map: Tensor,
        target_feature_map: Tensor,
        update_feature_map: Tensor,
        grid: Tensor,
        forward_trajectory: Optional[Tensor] = None,
        visibility: Optional[Tensor] = None,
        upsample_feature: bool = True
    ) -> Tensor:
        """
        Use trajectory consistency to determine visibility.
        Computes forward and backward trajectories and compares their distances.
        
        Args:
            query_points: Original query points (B, 3) containing (x, y, t)
            target_points: Predicted points at current time step (B, 2) containing (x, y)  
            time_step: Current time step
            query_feature_map: Query feature map (T, C, H, W)
            target_feature_map: Target feature map (T, C, H, W)
            update_feature_map: Update feature map (T, C, H, W)
            grid: Coordinate grid (H, W, 2)
            forward_trajectory: Optional forward trajectory (B, T, 2) for efficiency
            visibility: Optional current visibility tensor (B, T) for adaptive search radius
            
        Returns:
            current_visibility: Boolean tensor indicating visibility (B,)
        """
        query_frames = query_points[:, 2].long()  # (B,)
        
        # Initialize visibility for backward tracking
        try:
            visibility[:, time_step] = True
        except Exception as e:
            print(e)
        # Compute backward trajectory
        backward_trajectory = self.compute_backward_trajectory(
            query_points, target_points, time_step,
            query_feature_map, target_feature_map, update_feature_map, grid,
            visibility=visibility,
            upsample_feature=upsample_feature
        )
        
        # Compare trajectories and determine visibility
        current_visibility = self.compare_trajectories(
            forward_trajectory, backward_trajectory, 
            query_frames, time_step, visibility
        )
        
        # Free backward trajectory memory immediately
        del backward_trajectory
        
        return current_visibility
       
    def compute_backward_trajectory(
        self,
        query_points: Tensor,
        target_points: Tensor,
        time_step: int, 
        query_feature_map: Tensor,
        target_feature_map: Tensor,
        update_feature_map: Tensor,
        grid: Tensor,
        visibility: Optional[Tensor],
        upsample_feature: bool = True
    ) -> Tensor:
        """
        Compute backward trajectory from target points to query points (vectorized version).
        
        Args:
            query_points: Original query points (B, 3) containing (x, y, t)
            target_points: Predicted points at current time step (B, 2) containing (x, y)  
            time_step: Current time step
            query_feature_map: Query feature map (T, C, H, W)
            target_feature_map: Target feature map (T, C, H, W)
            update_feature_map: Update feature map (T, C, H, W)
            grid: Coordinate grid (H, W, 2)
            grid_size: Grid size for coordinate conversion
            visibility: visibility tensor (B, T) for adaptive search radius
        
        Returns:
            trajectory: (B, T, 2) trajectory from current time to query time
        """
        B = target_points.shape[0]
        T = query_feature_map.shape[0]
        query_frames = query_points[:, 2].long()  # (B,)
        
        # Initialize trajectory and features
        trajectory = torch.zeros((B, T, 2), device=self.device, dtype=query_points.dtype)
        trajectory[:, time_step, :] = target_points
        
        # Sample target features
        target_points_3d = torch.cat([
            target_points,
            torch.full((B, 1), time_step, device=self.device, dtype=query_feature_map.dtype)
        ], dim=1)
        target_feature = self.sample_feature(query_feature_map, target_points_3d, T)  # (B, C)
        current_positions = target_points.clone()  # (B, 2)
        
        # Get minimum start time to determine tracking range
        min_start_time = query_frames.min().item()
        
        # Track backward from time_step-1 to min_start_time
        for t in range(time_step - 1, min_start_time - 1, -1):
            active_mask = query_frames <= t  # (B,)
            if not active_mask.any():
                continue
            
            # Get active points data
            active_features = target_feature[active_mask]  # (N_active, C)
            active_positions = current_positions[active_mask]  # (N_active, 2)
            
            # Get correlation maps for all active points
            corr_maps = self.get_corr_maps(active_features, target_feature_map[t])  # (N_active, H, W)
            if not upsample_feature:
                corr_maps = F.interpolate(corr_maps.unsqueeze(1), size=(self.video_h, self.video_w), mode="bilinear").squeeze(1)
            corr_maps = self.softmax_corr_map(corr_maps)  # (N_active, H, W)
            
            # Apply search radius constraint based on visibility
            next_visibility = visibility[active_mask, t + 1]  # (N_active,)
            search_radius = torch.full(next_visibility.shape, self.search_radius, dtype=torch.float32, device=self.device)
            search_radius[~next_visibility] = float("inf")

            corr_maps = self.apply_mask(corr_maps, active_positions, grid, search_radius)
            
            # Predict previous positions
            prev_positions = self.soft_argmax(corr_maps, grid)  # (N_active, 2)
            trajectory[active_mask, t, :] = prev_positions
            
            # Update current positions for active points
            current_positions[active_mask] = prev_positions
            
            # Update features
            if update_feature_map is not None:
                target_feature[active_mask] = self.ema_update_query_feature(
                    target_feature[active_mask], 
                    update_feature_map, 
                    prev_positions, 
                    visibility[active_mask, t], 
                    t
                )
        
        return trajectory
    
    def compare_trajectories(
        self,
        forward_trajectory: Tensor,
        backward_trajectory: Tensor,
        query_frames: Tensor,
        time_step: int,
        visibility: Tensor
    ) -> Tensor:
        """
        Compare forward and backward trajectories to determine visibility (fully vectorized version).
        Only considers visible points for trajectory comparison.
        
        Args:
            forward_trajectory: (B, T, 2) forward trajectory
            backward_trajectory: (B, T, 2) backward trajectory
            query_frames: (B,) original time steps for each point
            time_step: Current time step
            visibility: (B, T) visibility tensor indicating which points are visible at each time step
            
        Returns:
            visibility: (B,) boolean tensor indicating visibility
        """
        B, T, _ = forward_trajectory.shape
        
        # Calculate squared differences for all trajectories
        diff_squared = torch.sum((forward_trajectory - backward_trajectory) ** 2, dim=2)  # (B, T)
        
        # Create mask for valid time ranges for each point
        time_indices = torch.arange(T, device=self.device)  # (T,)
        start_times = query_frames.unsqueeze(1)  # (B, 1)
        time_mask = (time_indices >= start_times) & (time_indices < time_step)  # (B, T)
        
        # Combine time mask with visibility mask - only consider visible points for comparison
        valid_mask = time_mask & visibility  # (B, T)
        
        # Apply mask and calculate mean squared distances
        masked_diff_squared = diff_squared * valid_mask.float()  # (B, T)
        sum_squared_distances = masked_diff_squared.sum(dim=1)  # (B,)
        valid_counts = valid_mask.sum(dim=1).float()  # (B,)
        mean_squared_distances = sum_squared_distances / valid_counts  # (B,)
        mean_trajectory_distances = torch.sqrt(mean_squared_distances)  # (B,)
        current_visibility = mean_trajectory_distances <= self.vis_threshold  # (B,)
        
        return current_visibility
        
    def initialize_tracking(self, query_points: Tensor, num_frames: int) -> tuple[Tensor, Tensor]:
        """
        query_points: Tensor of shape (B, 3) containing (x, y, t) coordinates
        """
        B, _ = query_points.shape
        trajectory = torch.zeros((B, num_frames, 2), device=self.device, dtype=query_points.dtype)
        visibility = torch.zeros((B, num_frames), dtype=torch.bool, device=self.device)
        start_indices = query_points[:, 2].long()
        trajectory[torch.arange(B), start_indices, :] = query_points[:, :2]
        visibility[torch.arange(B), start_indices] = True
        return trajectory, visibility

    def soft_argmax(self, corr: Tensor, grid: Tensor) -> Tensor:
        """
        corr: (B, H, W)
        grid: (H, W, 2)
        """
        # Find argmax positions
        B, H, W = corr.shape
        argmax = torch.argmax(rearrange(corr, "b h w -> b (h w)"), dim=-1)  # (B,)
        x_idx, y_idx = argmax % W, argmax // W  # (B,)
        argmax_coord = torch.stack((x_idx, y_idx), dim=-1)  # (B, 2)
        
        # Apply argmax mask
        corr = self.apply_mask(corr, argmax_coord.float(), grid, self.argmax_radius) # (B, H, W)
        
        # Normalize correlation maps
        corr_sum = torch.sum(corr, dim=(1, 2))  # (B,)
        corr = corr / corr_sum.unsqueeze(-1).unsqueeze(-1)  # (B, H, W)
        
        # Calculate target points using soft argmax
        target_point = torch.sum(grid.unsqueeze(0) * corr.unsqueeze(-1), dim=(1, 2))  # (B, 2)
        return target_point
    
    def apply_mask(self, corr: Tensor, coords: Tensor, grid: Tensor, radius: float | Tensor, default_value: float = 0.0) -> Tensor:
        """
        corr: (B, H, W)
        coords: (B, 2)
        grid: (H, W, 2)
        radius: radius of the mask or (B,)
        default_value: float
        """
        if isinstance(radius, float):
            radius = radius * torch.ones(corr.shape[0], dtype=corr.dtype, device=corr.device) # (B,)
        elif isinstance(radius, Tensor): pass
        else: raise ValueError(f"Invalid radius type: {type(radius)}")
        radius = radius.unsqueeze(-1).unsqueeze(-1) # (B, 1, 1)
        x = coords.reshape(-1, 1, 1, 2) # (B, 1, 1, 2)
        y = grid.unsqueeze(0) # (1, H, W, 2)
        # Use float32 for numerical stability in distance computation
        mask = torch.norm((x - y).to(torch.float32), dim=-1) <= radius.to(torch.float32)  # (B, H, W)
        corr = torch.where(mask, corr, torch.tensor(default_value, device=corr.device, dtype=corr.dtype))
        return corr
    
    def head_track(
        self,
        step: int,
        layer: int,
        head: int,
        query_points: Tensor, 
        query_feature_type: Literal["query", "key", "hidden_states"], 
        target_feature_type: Literal["query", "key", "hidden_states"],
        update_feature_type: Literal["query", "key", "hidden_states", None] = None,
        output_dir: str = None,
        upsample_feature: bool = True,
    ):
        # Get query feature
        num_chunk, chunk_size = self.get_num_chunk()
        query_points = query_points.to(self.device)
        query_feature_map = self.load_feature(query_feature_type, step, layer, head, 0)  # (T, C, H, W)
        T = query_feature_map.shape[0]
        if upsample_feature:
            query_feature_map = self.upsample_feature(query_feature_map, T)
        query_feature = self.sample_feature(query_feature_map, query_points, chunk_size) # (B, C)
        updated_query_feature = query_feature.clone()  # (B, C)
        # Initialize tracking
        try:
            grid = get_grid(self.video_h, self.video_w).to(self.device) # (H, W, 2)
        except Exception as e:
            print(f"Error getting grid: {e}")
            return
        trajectory, visibility = self.initialize_tracking(query_points, self.frames)

        
        for chunk_id in range(num_chunk):
            # Load features
            query_feature_map = self.load_feature(query_feature_type, step, layer, head, chunk_id)  # (T, C, H, W)
            target_feature_map = self.load_feature(target_feature_type, step, layer, head, chunk_id)  # (T, C, H, W)
            frames_in_chunk = query_feature_map.shape[0] if chunk_id == 0 else query_feature_map.shape[0] + 1
            if update_feature_type is not None:
                update_feature_map = self.load_feature(update_feature_type, step, layer, head, chunk_id)  # (T, C, H, W)
            
            if upsample_feature:
                query_feature_map = self.upsample_feature(query_feature_map, frames_in_chunk)
                target_feature_map = self.upsample_feature(target_feature_map, frames_in_chunk)
                if update_feature_type is not None:
                    update_feature_map = self.upsample_feature(update_feature_map, frames_in_chunk)
                else:
                    update_feature_map = None
            else:
                query_feature_map = self.interpolate_feature(query_feature_map, frames_in_chunk)
                target_feature_map = self.interpolate_feature(target_feature_map, frames_in_chunk)
                if update_feature_type is not None:
                    update_feature_map = self.interpolate_feature(update_feature_map, frames_in_chunk)
                else:
                    update_feature_map = None

            if chunk_id == 0:
                chunk_trajectory = trajectory[:, : chunk_size]
                chunk_visibility = visibility[:, : chunk_size]
            elif chunk_id == num_chunk - 1:
                chunk_trajectory = trajectory[:, chunk_id * chunk_size - 1 :]
                chunk_visibility = visibility[:, chunk_id * chunk_size - 1 :]
            else:
                chunk_trajectory = trajectory[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size]
                chunk_visibility = visibility[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size]

            # Track
            for i in range(1, frames_in_chunk):
                # Get correlation maps
                corr = self.get_corr_maps(updated_query_feature, target_feature_map[i])  # (N, H, W)
                corr = self.softmax_corr_map(corr)  # (N, H, W)
                if not upsample_feature:
                    corr = F.interpolate(corr.unsqueeze(1), size=(self.video_h, self.video_w), mode="bilinear").squeeze(1)
                
                # Apply search mask
                prev_coords = chunk_trajectory[:, i-1, :]  # (N, 2)
                prev_visibility = chunk_visibility[:, i-1] # (N,)
                search_radius = torch.full(prev_visibility.shape, self.search_radius, dtype=torch.float32, device=self.device) # (N,)
                search_radius[~prev_visibility] = float("inf")
                corr = self.apply_mask(corr, prev_coords, grid, search_radius)

                # Apply search mask
                prev_coords = chunk_trajectory[:, i-1, :]  # (N, 2)
                prev_visibility = chunk_visibility[:, i-1] # (N,)
                search_radius = torch.full(prev_visibility.shape, self.search_radius, dtype=torch.float32, device=self.device) # (N,)
                search_radius[~prev_visibility] = float("inf")
                corr = self.apply_mask(corr, prev_coords, grid, search_radius)
                
                # Update trajectory and visibility
                target_point = self.soft_argmax(corr, grid) # (N, 2)
                chunk_visibility[:, i] = self.get_visibility(
                    query_points, 
                    target_point, 
                    i, 
                    query_feature_map,
                    target_feature_map,
                    update_feature_map,
                    grid,
                    forward_trajectory=chunk_trajectory,
                    visibility=chunk_visibility,
                    upsample_feature=upsample_feature
                ) # (N,)
                chunk_trajectory[:, i] = target_point

                # Update query features
                if update_feature_map is not None:
                    updated_query_feature = self.ema_update_query_feature(
                        updated_query_feature, 
                        update_feature_map, 
                        target_point, 
                        chunk_visibility[:, i], 
                        i
                    )
            
            if chunk_id == 0:
                trajectory[:, : chunk_size] = chunk_trajectory
                visibility[:, : chunk_size] = chunk_visibility
            elif chunk_id == num_chunk - 1:
                trajectory[:, chunk_id * chunk_size - 1 :] = chunk_trajectory
                visibility[:, chunk_id * chunk_size - 1 :] = chunk_visibility
            else:
                trajectory[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size] = chunk_trajectory
                visibility[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size] = chunk_visibility
            
            del query_feature_map, target_feature_map, chunk_trajectory, chunk_visibility
            if update_feature_map is not None:
                del update_feature_map
            torch.cuda.empty_cache()
        
        # Final cleanup before saving
        del query_feature, updated_query_feature, grid
        torch.cuda.empty_cache()
        self.save_track(trajectory, visibility, step, layer, head, query_feature_type, target_feature_type, output_dir)

    def layer_track(
        self,
        step: int,
        layer: int,
        query_points: Tensor, 
        query_feature_type: Literal["query", "key", "hidden_states"], 
        target_feature_type: Literal["query", "key", "hidden_states"],
        update_feature_type: Literal["query", "key", "hidden_states", None] = None,
        output_dir: str = None,
        upsample_feature: bool = True,
    ):
        # Get query feature
        num_chunk, chunk_size = self.get_num_chunk()
        query_points = query_points.to(self.device)
        query_feature_maps = []
        for head in range(12):
            query_feature_map = self.load_feature(query_feature_type, step, layer, head, 0)  # (T, C, H, W)
            query_feature_maps.append(query_feature_map)
        query_feature_map = torch.cat(query_feature_maps, dim=1) # (T, head*C, H, W)
        T = query_feature_map.shape[0]
        if upsample_feature:
            query_feature_map = self.upsample_feature(query_feature_map, T)
        query_feature = self.sample_feature(query_feature_map, query_points, chunk_size) # (B, C)
        updated_query_feature = query_feature.clone()  # (B, C)
        # Initialize tracking
        grid = get_grid(self.video_h, self.video_w).to(self.device) # (H, W, 2)
        trajectory, visibility = self.initialize_tracking(query_points, self.frames)

        for chunk_id in range(num_chunk):
            # Load features
            query_feature_maps = []
            target_feature_maps = []
            for head in range(12):
                query_feature_map = self.load_feature(query_feature_type, step, layer, head, chunk_id)  # (T, C, H, W)
                target_feature_map = self.load_feature(target_feature_type, step, layer, head, chunk_id)  # (T, C, H, W)
                query_feature_maps.append(query_feature_map)
                target_feature_maps.append(target_feature_map)
            query_feature_map = torch.cat(query_feature_maps, dim=1) # (T, head*C, H, W)
            target_feature_map = torch.cat(target_feature_maps, dim=1) # (T, head*C, H, W)
            frames_in_chunk = query_feature_map.shape[0] if chunk_id == 0 else query_feature_map.shape[0] + 1
            if update_feature_type is not None:
                update_feature_maps = []
                for head in range(12):
                    update_feature_map = self.load_feature(update_feature_type, step, layer, head, chunk_id)  # (T, C, H, W)
                    update_feature_maps.append(update_feature_map)
                update_feature_map = torch.cat(update_feature_maps, dim=1) # (T, head*C, H, W)
            
            if upsample_feature:
                query_feature_map = self.upsample_feature(query_feature_map, frames_in_chunk)
                target_feature_map = self.upsample_feature(target_feature_map, frames_in_chunk)
                if update_feature_type is not None:
                    update_feature_map = self.upsample_feature(update_feature_map, frames_in_chunk)
                else:
                    update_feature_map = None
            else:
                query_feature_map = self.interpolate_feature(query_feature_map, frames_in_chunk)
                target_feature_map = self.interpolate_feature(target_feature_map, frames_in_chunk)
                if update_feature_type is not None:
                    update_feature_map = self.interpolate_feature(update_feature_map, frames_in_chunk)
                else:
                    update_feature_map = None
            if chunk_id == 0:
                chunk_trajectory = trajectory[:, : chunk_size]
                chunk_visibility = visibility[:, : chunk_size]
            elif chunk_id == num_chunk - 1:
                chunk_trajectory = trajectory[:, chunk_id * chunk_size - 1 :]
                chunk_visibility = visibility[:, chunk_id * chunk_size - 1 :]
            else:
                chunk_trajectory = trajectory[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size]
                chunk_visibility = visibility[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size]

            # Track
            for i in range(1, frames_in_chunk):
                # Get correlation maps
                corr = self.get_corr_maps(updated_query_feature, target_feature_map[i])  # (N, H, W)
                corr = self.softmax_corr_map(corr)  # (N, H, W)
                if not upsample_feature:
                    corr = F.interpolate(corr.unsqueeze(1), size=(self.video_h, self.video_w), mode="bilinear").squeeze(1)
                
                # Update trajectory and visibility
                target_point = self.soft_argmax(corr, grid) # (N, 2)
                chunk_visibility[:, i] = self.get_visibility(
                    query_points, 
                    target_point, 
                    i, 
                    query_feature_map,
                    target_feature_map,
                    update_feature_map,
                    grid,
                    forward_trajectory=chunk_trajectory,
                    visibility=chunk_visibility,
                    upsample_feature=upsample_feature
                ) # (N,)
                chunk_trajectory[:, i] = target_point

                # Update query features
                if update_feature_map is not None:
                    updated_query_feature = self.ema_update_query_feature(
                        updated_query_feature, 
                        update_feature_map, 
                        target_point, 
                        chunk_visibility[:, i], 
                        i
                    )
            
            if chunk_id == 0:
                trajectory[:, : chunk_size] = chunk_trajectory
                visibility[:, : chunk_size] = chunk_visibility
            elif chunk_id == num_chunk - 1:
                trajectory[:, chunk_id * chunk_size - 1 :] = chunk_trajectory
                visibility[:, chunk_id * chunk_size - 1 :] = chunk_visibility
            else:
                trajectory[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size] = chunk_trajectory
                visibility[:, chunk_id * chunk_size - 1 : (chunk_id + 1) * chunk_size] = chunk_visibility
            
            del query_feature_map, target_feature_map, chunk_trajectory, chunk_visibility
            if update_feature_map is not None:
                del update_feature_map
            torch.cuda.empty_cache()
        
        # Final cleanup before saving
        del query_feature, updated_query_feature, grid
        torch.cuda.empty_cache()
        self.save_track(trajectory, visibility, step, layer, -1, query_feature_type, target_feature_type, output_dir)

    def save_track(
        self, 
        trajectory: Tensor, 
        visibility: Tensor, 
        step: int, 
        layer: int, 
        head: int, 
        query_feature_type: Literal["query", "key", "hidden_states"], 
        target_feature_type: Literal["query", "key", "hidden_states"],
        output_dir: str = None,
    ):  
        sub_dir = query_feature_type + "-" + target_feature_type     
        if head == -1:
            output_dir = self.layer_track_dir if output_dir is None else Path(output_dir)/"layer_track"
            track_path = output_dir / sub_dir / \
                f"step_{step:03d}" / f"layer_{layer:02d}_track.pt"
            visibility_path = output_dir / sub_dir / \
                f"step_{step:03d}" / f"layer_{layer:02d}_visibility.pt" 
        else:
            output_dir = self.head_track_dir if output_dir is None else Path(output_dir)/"head_track"
            track_path = output_dir / sub_dir / \
                f"step_{step:03d}" / f"layer_{layer:02d}" / f"head_{head:02d}_track.pt"
            visibility_path = output_dir / sub_dir / \
                f"step_{step:03d}" / f"layer_{layer:02d}" / f"head_{head:02d}_visibility.pt" 

        Path(track_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(trajectory, track_path)
        Path(visibility_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(visibility, visibility_path)

    