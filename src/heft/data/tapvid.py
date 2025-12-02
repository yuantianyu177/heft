import pickle
import cv2
import numpy as np
import torch
import random
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Union
from ..utils.io import save_video


class TapVid:   
    def __init__(self, root: str = "datasets/tapvid_davis"):
        self._load_data(Path(root))
        
    def _load_data(self, root: Path):          
        with open(root/"tapvid_davis.pkl", 'rb') as f:
            self.data = pickle.load(f)
        self.video_names = sorted(list(self.data.keys()))
        
    def get_tracks(
        self,
        output_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get tracks and visibility data.
        Only returns tracks that are visible from the first frame (first_visible_indices=0).
        
        Args:
            output_size (tuple, optional): Target size (height, width) to resize video. Default: None
        
        Returns:
            tracks_list: List[np.ndarray] of shape (num_points, num_frames, 2) containing normalized (x, y) coordinates [0-1]
            visibility_list: List[np.ndarray] of shape (num_points, num_frames) containing visibility flags
        """ 
        tracks_list = []
        visibility_list = []
        for name in self.video_names:
            video_data = self.data[name]
            tracks = video_data['points']
            visibility = ~video_data['occluded']  # Invert occluded to get visibility
            
            # Only keep tracks that are visible from the first frame
            first_visible_indices = np.argmax(visibility, axis=1)  # Shape: (num_points,)
            first_frame_mask = first_visible_indices == 0  # Shape: (num_points,)
            tracks = tracks[first_frame_mask]
            visibility = visibility[first_frame_mask]
            
            if output_size is not None:
                tracks[:, :, 0] = tracks[:, :, 0] * (output_size[1] - 1)
                tracks[:, :, 1] = tracks[:, :, 1] * (output_size[0] - 1)
            tracks_list.append(tracks)
            visibility_list.append(visibility)
            
        return tracks_list, visibility_list   

    def get_query_points(
        self,
        output_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Get query points for all videos using vectorized operations.
        Only returns points whose query frame is the first frame (first_visible_indices=0).
        
        Args:
            output_size (tuple, optional): Target size (height, width) to resize coordinates. 
                                         If None, returns normalized coordinates [0-1].
        
        Returns:
            List[np.ndarray]: List of query points for each video.
                            Each array has shape (num_points, 3) where 3 represents (x, y, t).
                            Query point is the first visible point of each track.
        """
        query_points_list = []
        
        for name in self.video_names:
            video_data = self.data[name]
            tracks = video_data['points']  # Shape: (num_points, num_frames, 2)
            visibility = ~video_data['occluded']  # Shape: (num_points, num_frames)
            
            first_visible_indices = np.argmax(visibility, axis=1)  # Shape: (num_points,)
            
            # Only keep points that are visible from the first frame
            first_frame_mask = first_visible_indices == 0  # Shape: (num_points,)
            first_visible_indices = first_visible_indices[first_frame_mask]
            tracks_filtered = tracks[first_frame_mask]
        
            query_coordinates = tracks_filtered[np.arange(len(first_visible_indices)), first_visible_indices]  # Shape: (num_points, 2)
            query_points = np.column_stack([
                query_coordinates[:, 0],  # x coordinates
                query_coordinates[:, 1],  # y coordinates  
                first_visible_indices.astype(np.float32)  # time indices
            ])
            
            # Apply output_size scaling if provided
            if output_size is not None:
                query_points[:, 0] *= (output_size[1] - 1)  # scale x by width
                query_points[:, 1] *= (output_size[0] - 1) # scale y by height
            
            query_points_list.append(query_points.astype(np.float32))
        
        return query_points_list

    def get_scene_names(self) -> List[str]:
        return self.video_names.copy()
    
    def save_videos(
        self,
        output_size: Tuple[int, int] = (480, 832),
        output_dir: str = "."
    ):
        for i, name in enumerate(self.video_names):
            video_data = self.data[name]
            video = video_data['video']
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_video(video, f"{output_dir}/{name}.mp4", output_size=output_size)


    def __len__(self) -> int:
        return len(self.video_names)

class TapVidRGBStack:   
    def __init__(self, root: str = "datasets/tapvid_davis"):
        self._load_data(Path(root))
        
    def _load_data(self, root: Path):          
        with open(root/"tapvid_rgb_stacking.pkl", 'rb') as f:
            self.data = pickle.load(f)
        
    def get_tracks(
        self, 
        output_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get tracks and visibility data.
        Only returns tracks that are visible from the first frame (first_visible_indices=0).
        
        Args:
            output_size (tuple, optional): Target size (height, width) to resize video. Default: None
        
        Returns:
            tracks_list: List[np.ndarray] of shape (num_points, num_frames, 2) containing normalized (x, y) coordinates [0-1]
            visibility_list: List[np.ndarray] of shape (num_points, num_frames) containing visibility flags
        """ 
        tracks_list = []
        visibility_list = []
        for video_data in self.data:
            tracks = video_data['points']
            visibility = ~video_data['occluded']  # Invert occluded to get visibility
            
            # Only keep tracks that are visible from the first frame
            first_visible_indices = np.argmax(visibility, axis=1)  # Shape: (num_points,)
            first_frame_mask = first_visible_indices == 0  # Shape: (num_points,)
            tracks = tracks[first_frame_mask]
            visibility = visibility[first_frame_mask]
            
            if output_size is not None:
                tracks[:, :, 0] = tracks[:, :, 0] * (output_size[1] - 1)
                tracks[:, :, 1] = tracks[:, :, 1] * (output_size[0] - 1)
            tracks_list.append(tracks)
            visibility_list.append(visibility)
            
        return tracks_list, visibility_list   
    

    def get_query_points(
        self,
        output_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Get query points for all videos using vectorized operations.
        Only returns points whose query frame is the first frame (first_visible_indices=0).
        
        Args:
            output_size (tuple, optional): Target size (height, width) to resize coordinates. 
                                         If None, returns normalized coordinates [0-1].
        
        Returns:
            List[np.ndarray]: List of query points for each video.
                            Each array has shape (num_points, 3) where 3 represents (x, y, t).
                            Query point is the first visible point of each track.
        """
        query_points_list = []
        
        for video_data in self.data:
            tracks = video_data['points']  # Shape: (num_points, num_frames, 2)
            visibility = ~video_data['occluded']  # Shape: (num_points, num_frames)
            
            first_visible_indices = np.argmax(visibility, axis=1)  # Shape: (num_points,)
            
            # Only keep points that are visible from the first frame
            first_frame_mask = first_visible_indices == 0  # Shape: (num_points,)
            first_visible_indices = first_visible_indices[first_frame_mask]
            tracks_filtered = tracks[first_frame_mask]
        
            query_coordinates = tracks_filtered[np.arange(len(first_visible_indices)), first_visible_indices]  # Shape: (num_points, 2)
            query_points = np.column_stack([
                query_coordinates[:, 0],  # x coordinates
                query_coordinates[:, 1],  # y coordinates  
                first_visible_indices.astype(np.float32)  # time indices
            ])
            
            # Apply output_size scaling if provided
            if output_size is not None:
                query_points[:, 0] *= (output_size[1] - 1)  # scale x by width
                query_points[:, 1] *= (output_size[0] - 1) # scale y by height
            
            query_points_list.append(query_points.astype(np.float32))
        
        return query_points_list

    
    def save_videos(
        self,
        output_size: Tuple[int, int] = (480, 832),
        output_dir: str = "."
    ):
        for i, video_data in enumerate(self.data):
            video = video_data['video']
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_video(video, f"{output_dir}/{i:04d}.mp4", output_size=output_size)

    def get_scene_names(self) -> List[str]:
        return [f"{i:04d}" for i in range(len(self.data))]

    def __len__(self) -> int:
        return len(self.data)


class TapVidKinetics: 
    error_list = [62,65,202,301,350,364,396,413,414,421,469,475,527,543,566,596,703,807,844,892,960,1010,1026,1044]
    def __init__(self, root: str):
        self._load_data(Path(root))
        
    def _load_data(self, root: Path):
        self.data = []
        for p in sorted(root.rglob("*.pkl"), key=lambda x: x.name):
            print(f"loading {p.name}...")
            with open(p, 'rb') as f:
                self.data.extend(pickle.load(f))
        # Filter out error indices
        self.data = [data for i, data in enumerate(self.data) if i not in self.error_list]
        # Sample 30 elements with fixed seed for reproducibility
        if len(self.data) > 30:
            random.seed(42)
            self.data = random.sample(self.data, 30)
        
    def get_tracks(
        self, 
        output_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Get tracks and visibility data.
        Only returns tracks that are visible from the first frame (first_visible_indices=0).
        
        Args:
            output_size (tuple, optional): Target size (height, width) to resize video. Default: None
        
        Returns:
            tracks_list: List[np.ndarray] of shape (num_points, num_frames, 2) containing normalized (x, y) coordinates [0-1]
            visibility_list: List[np.ndarray] of shape (num_points, num_frames) containing visibility flags
        """ 
        tracks_list = []
        visibility_list = []
        for video_data in self.data:
            tracks = video_data['points']
            visibility = ~video_data['occluded']  # Invert occluded to get visibility
            
            # Only keep tracks that are visible from the first frame
            first_visible_indices = np.argmax(visibility, axis=1)  # Shape: (num_points,)
            first_frame_mask = first_visible_indices == 0  # Shape: (num_points,)
            tracks = tracks[first_frame_mask]
            visibility = visibility[first_frame_mask]
            
            if output_size is not None:
                tracks[:, :, 0] = tracks[:, :, 0] * (output_size[1] - 1)
                tracks[:, :, 1] = tracks[:, :, 1] * (output_size[0] - 1)
            tracks_list.append(tracks)
            visibility_list.append(visibility)
            
        return tracks_list, visibility_list   
    

    def get_query_points(
        self,
        output_size: Optional[Tuple[int, int]] = None
    ) -> List[np.ndarray]:
        """
        Get query points for all videos using vectorized operations.
        Only returns points whose query frame is the first frame (first_visible_indices=0).
        
        Args:
            output_size (tuple, optional): Target size (height, width) to resize coordinates. 
                                         If None, returns normalized coordinates [0-1].
        
        Returns:
            List[np.ndarray]: List of query points for each video.
                            Each array has shape (num_points, 3) where 3 represents (x, y, t).
                            Query point is the first visible point of each track.
        """
        query_points_list = []
        
        for video_data in self.data:
            tracks = video_data['points']  # Shape: (num_points, num_frames, 2)
            visibility = ~video_data['occluded']  # Shape: (num_points, num_frames)
            
            first_visible_indices = np.argmax(visibility, axis=1)  # Shape: (num_points,)
            
            # Only keep points that are visible from the first frame
            first_frame_mask = first_visible_indices == 0  # Shape: (num_points,)
            first_visible_indices = first_visible_indices[first_frame_mask]
            tracks_filtered = tracks[first_frame_mask]
        
            query_coordinates = tracks_filtered[np.arange(len(first_visible_indices)), first_visible_indices]  # Shape: (num_points, 2)
            query_points = np.column_stack([
                query_coordinates[:, 0],  # x coordinates
                query_coordinates[:, 1],  # y coordinates  
                first_visible_indices.astype(np.float32)  # time indices
            ])

            # Apply output_size scaling if provided
            if output_size is not None:
                query_points[:, 0] *= (output_size[1] - 1)  # scale x by width
                query_points[:, 1] *= (output_size[0] - 1) # scale y by height
            
            query_points_list.append(query_points.astype(np.float32))
        
        return query_points_list
    
    
    def save_videos(
        self,
        output_size: Tuple[int, int] = (480, 832),
        output_dir: str = "."
    ):
        for i, video_data in enumerate(self.data):
            video = preprocess_video_bytes(video_data['video'])
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            save_video(video, f"{output_dir}/{i:04d}.mp4", output_size=output_size)

    def get_scene_names(self) -> List[str]:
        return [f"{i:04d}" for i in range(len(self.data))]

    def __len__(self) -> int:
        return len(self.data)
  

def preprocess_video_bytes(video_bytes):
    """
    Convert array of JPEG bytes to numpy array of shape (T, H, W, C) uint8.
    """
    frames = []
    for b in video_bytes:
        frame = cv2.imdecode(np.frombuffer(b, dtype=np.uint8), cv2.IMREAD_COLOR)  # BGR
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转成 RGB
        frames.append(frame)
    return np.stack(frames)  # (T, H, W, C)