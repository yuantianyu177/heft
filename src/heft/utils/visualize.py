"""Visualization utilities for track rainbow trails."""

import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections
import cv2
from typing import Optional, Tuple, List
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from .io import load_video, save_video


def get_rainbow_colors(num_colors: int) -> np.ndarray:
    """Generate rainbow colors for visualization.
    
    Args:
        num_colors: Number of colors to generate
        
    Returns:
        colors: Array of shape [num_colors, 3] with RGB values in [0, 1]
    """
    colors = []
    for i in range(num_colors):
        hue = i / max(num_colors, 1)
        saturation = 0.9 + np.random.rand() * 0.1
        lightness = 0.5 + np.random.rand() * 0.1
        color = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(color)
    return np.array(colors)


def smooth_tracks(tracks: np.ndarray, visibility: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth tracks using Gaussian filtering.
    
    Args:
        tracks: Track positions of shape [N, T, 2] in (x, y) format
        visibility: Visibility mask of shape [N, T]
        sigma: Gaussian smoothing sigma. Higher values = more smoothing
        
    Returns:
        smoothed_tracks: Smoothed track positions of shape [N, T, 2]
    """
    if sigma <= 0:
        return tracks
    
    N, T, _ = tracks.shape
    smoothed_tracks = np.zeros_like(tracks)
    
    for n in range(N):
        # Only smooth visible segments
        visible_mask = visibility[n]
        
        if np.sum(visible_mask) < 2:
            # Not enough visible points to smooth
            smoothed_tracks[n] = tracks[n]
            continue
        
        # Smooth x and y coordinates separately
        smoothed_tracks[n, :, 0] = gaussian_filter1d(
            tracks[n, :, 0], sigma=sigma, mode='nearest'
        )
        smoothed_tracks[n, :, 1] = gaussian_filter1d(
            tracks[n, :, 1], sigma=sigma, mode='nearest'
        )
        
        # Keep original positions for invisible points
        smoothed_tracks[n, ~visible_mask] = tracks[n, ~visible_mask]
    
    return smoothed_tracks


def plot_rainbow_tracks(
    video: np.ndarray,
    tracks: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    point_size: int = 3,
    point_marker: str = 'circle',
    trail_length: int = 10,
    line_thickness: int = 2,
    fade_trail: bool = True,
    max_tracks: int = 1000,
    smooth_sigma: float = 0.0,
    bg_tracks: Optional[np.ndarray] = None,
    bg_visibility: Optional[np.ndarray] = None
) -> np.ndarray:
    """Plot tracks with rainbow trails using optimized matplotlib.
    
    Args:
        video: Video frames of shape [T, H, W, 3], uint8 [0, 255]
        tracks: Track positions of shape [N, T, 2] in (x, y) format  
        visibility: Visibility mask of shape [N, T]. If None, all visible
        point_size: Size of track points
        point_marker: Shape of point markers ('circle' or 'diamond')
        trail_length: Length of trail in frames
        line_thickness: Thickness of trail lines
        fade_trail: Whether to fade trail over time
        max_tracks: Maximum number of tracks to visualize
        smooth_sigma: Gaussian smoothing sigma (0 = no smoothing, higher = more smooth)
        bg_tracks: Background tracks of shape [M, T, 2] for camera motion compensation. Optional.
        bg_visibility: Background visibility of shape [M, T]. Optional.
        
    Returns:
        result_video: Video with rainbow tracks of shape [T, H, W, 3]
    """
    T, H, W, _ = video.shape
    N, _, _ = tracks.shape
    
    if visibility is None:
        visibility = np.ones((N, T), dtype=bool)
    
    # Keep original tracks for current frame points (not smoothed)
    original_tracks = tracks.copy()
    
    # Apply smoothing if requested (for trail visualization only)
    smoothed_tracks = None
    if smooth_sigma > 0:
        smoothed_tracks = smooth_tracks(tracks, visibility, smooth_sigma)
    
    # Sample tracks if too many
    if N > max_tracks:
        print(f"Sampling {max_tracks} tracks from {N} total tracks")
        # Sample tracks that are visible in most frames
        visibility_count = np.sum(visibility, axis=1)
        indices = np.argsort(visibility_count)[-max_tracks:]
        original_tracks = original_tracks[indices]
        if smoothed_tracks is not None:
            smoothed_tracks = smoothed_tracks[indices]
        visibility = visibility[indices]
        N = max_tracks
    
    # Generate rainbow colors for sampled tracks
    colors = get_rainbow_colors(N)
    
    # Convert point marker to matplotlib marker code
    marker_code = 'o' if point_marker == 'circle' else 'd'
    
    # Pre-compute camera displacement for each frame pair if bg_tracks provided
    # camera_displacement[t, s] = mean displacement from frame s to frame t
    camera_displacement = None
    if bg_tracks is not None:
        M = bg_tracks.shape[0]
        if bg_visibility is None:
            bg_visibility = np.ones((M, T), dtype=bool)
        
        print("Pre-computing camera motion compensation...")
        camera_displacement = np.zeros((T, T, 2), dtype=np.float32)
        for t in range(T):
            for s in range(T):
                if s != t:
                    # Calculate displacement from frame s to frame t
                    # diff = bg_tracks[t] - bg_tracks[s]
                    visible_mask = bg_visibility[:, t] & bg_visibility[:, s]
                    if np.any(visible_mask):
                        diff = bg_tracks[visible_mask, t] - bg_tracks[visible_mask, s]
                        camera_displacement[t, s] = np.mean(diff, axis=0)
    
    # Use lower DPI for faster rendering
    figure_dpi = 64
    
    print(f"Rendering rainbow tracks for {N} tracks...")
    result_frames = []
    
    for t in tqdm(range(T)):
        fig = plt.figure(
            figsize=(W / figure_dpi, H / figure_dpi),
            dpi=figure_dpi,
            frameon=False,
            facecolor='w'
        )
        ax = fig.add_subplot()
        ax.axis('off')
        # Set extent to align pixel coordinates: x from -0.5 to W-0.5, y from -0.5 to H-0.5
        # This ensures that pixel centers are at integer coordinates
        ax.imshow(video[t].astype(np.uint8), extent=[-0.5, W - 0.5, H - 0.5, -0.5])
        # Set axis limits to match image extent exactly
        ax.set_xlim(-0.5, W - 0.5)
        ax.set_ylim(H - 0.5, -0.5)
        
        # Collect all line segments and colors for batch rendering
        all_line_segments = []
        all_line_colors = []
        all_point_coords = []
        all_point_colors = []
        
        for n in range(N):
            if not visibility[n, t]:
                continue
                
            # Get trail points from current frame backwards
            trail_points = []
            
            for dt in range(min(trail_length, t + 1)):
                frame_idx = t - dt
                if frame_idx >= 0 and visibility[n, frame_idx]:
                    # Use original tracks for current frame (dt=0), smoothed for history (dt>0)
                    if dt == 0:
                        # Current frame: always use original (unsmoothed) coordinates
                        point = original_tracks[n, frame_idx].copy()
                    else:
                        # Historical frames: use smoothed tracks if available
                        if smoothed_tracks is not None:
                            point = smoothed_tracks[n, frame_idx].copy()
                        else:
                            point = original_tracks[n, frame_idx].copy()
                    
                    # Apply camera motion compensation for historical frames
                    # Current frame (dt=0) keeps original position
                    if camera_displacement is not None and dt > 0:
                        # Add camera displacement from frame_idx to current frame t
                        # This moves historical point to compensate for camera motion
                        point = point + camera_displacement[t, frame_idx]
                    
                    # Ensure point is within bounds
                    x = np.clip(point[0], 0, W - 1)
                    y = np.clip(point[1], 0, H - 1)
                    trail_points.append([x, y])
            
            # Calculate alpha values based on ACTUAL number of points
            # This ensures visible gradient regardless of trail_length parameter
            trail_alphas = []
            num_points = len(trail_points)
            if num_points > 0:
                for i in range(num_points):
                    if fade_trail:
                        # Linear fade from 1.0 (current) to 0.0 (oldest)
                        alpha = 1.0 - (i / max(num_points - 1, 1))
                    else:
                        alpha = 1.0
                    trail_alphas.append(alpha)
            
            # Create line segments for trail
            if len(trail_points) > 1:
                trail_points = np.array(trail_points)
                for i in range(len(trail_points) - 1):
                    all_line_segments.append(trail_points[i:i+2])
                    
                    # Use average alpha of the two endpoints for the line segment
                    alpha = (trail_alphas[i] + trail_alphas[i+1]) / 2.0
                    
                    color_with_alpha = list(colors[n]) + [alpha]
                    all_line_colors.append(color_with_alpha)
            
            # Add current point with its alpha value
            if len(trail_points) > 0:
                all_point_coords.append(trail_points[0])
                # Store color and alpha separately for points
                all_point_colors.append(list(colors[n]) + [trail_alphas[0]])
        
        # Batch render all line segments
        if all_line_segments:
            line_collection = matplotlib.collections.LineCollection(
                all_line_segments, 
                colors=all_line_colors, 
                linewidths=line_thickness
            )
            ax.add_collection(line_collection)
        
        # Render all points with individual alpha values
        if all_point_coords:
            all_point_coords = np.array(all_point_coords)
            all_point_colors = np.array(all_point_colors)
            
            # Extract RGB and alpha separately
            colors_rgb = all_point_colors[:, :3]
            alphas = all_point_colors[:, 3]
            
            # Draw points with individual alpha values
            # Group by similar alpha for efficiency
            unique_alphas = np.unique(np.round(alphas, 2))
            for alpha_val in unique_alphas:
                mask = np.abs(alphas - alpha_val) < 0.01
                if np.any(mask):
                    ax.scatter(
                        all_point_coords[mask, 0],
                        all_point_coords[mask, 1],
                        s=point_size * 10,
                        c=colors_rgb[mask],
                        marker=marker_code,
                        alpha=float(alpha_val),
                        edgecolors='none'
                    )
        
        # Convert matplotlib figure to numpy array
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        
        # Get image from canvas
        canvas_width, canvas_height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(
            int(canvas_height), int(canvas_width), 4
        )[:, :, :3]  # Remove alpha channel
        
        # Resize to match original video dimensions if needed
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        
        result_frames.append(img)
        plt.close(fig)
    
    return np.stack(result_frames, axis=0)