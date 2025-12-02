import torch
import numpy as np
from pathlib import Path
import imageio
import cv2
from typing import Literal


def _resize_frames(frames: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    resized_frames = []
    for frame in frames:
        h, w = target_size
        resized_frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LANCZOS4)
        resized_frames.append(resized_frame)
    return np.stack(resized_frames)


def load_video(
    video_path: str, 
    normalize: bool = False, 
    output_size: tuple[int, int] = None, 
    output_type: Literal["pt", "np"] = "pt"
) -> torch.Tensor | np.ndarray:
    """
    Load video and return as tensor of type output_type.
    
    Args:
        video_path (str): Path to the video file
        normalize (bool, optional): Whether to normalize pixel values to [0, 1]. 
                                   If False, returns values in [0, 255]. Default: False
        output_size (tuple[int, int], optional): Target size (height, width) to resize video. Default: None
        output_type (Literal["pt", "np"], optional): Type of the output tensor. Default: "pt"
    Returns:
        torch.Tensor | np.ndarray: Video tensor with shape (T, C, H, W) or (T, H, W, C) where T is number of frames,
                     C is number of channels, H is height, W is width
    """
    video = imageio.mimread(video_path, memtest=False)
    video = np.stack(video)

    if output_size is not None:
        video = _resize_frames(video, output_size)

    if normalize:
        video = video.astype(np.float32) / 255.0

    if output_type == "np":
        return video # (T, H, W, C)
    elif output_type == "pt":
        return torch.from_numpy(video).permute(0, 3, 1, 2) # (T, C, H, W)
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def save_video(
    video: np.ndarray | torch.Tensor, 
    save_path: str, 
    fps: int = 25, 
    output_size: tuple[int, int] = None
) -> None:
    """
    Save video array to file.
    
    Args:
        video (np.ndarray | torch.Tensor): Video array with shape (T, H, W, C) or (T, C, H, W), in [0, 255]
        save_path (str): Path where to save the video file
        fps (int, optional): Frames per second for the output video. Default: 25
        output_size (tuple[int, int], optional): Target size (height, width) to resize video. Default: None
    """ 
    if isinstance(video, torch.Tensor):
        video = video.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8) # (T, H, W, C)
    elif isinstance(video, np.ndarray):
        video = video.astype(np.uint8) # (T, H, W, C)
    else:
        raise ValueError(f"Invalid video type: {type(video)}")

    if output_size is not None:
        video = _resize_frames(video, output_size)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(save_path, video, fps=fps, macro_block_size=1)


def load_frames(
    frames_dir: str,
    normalize: bool = False,
    output_size: tuple[int, int] = None,
    output_type: Literal["pt", "np"] = "pt"
) -> torch.Tensor | np.ndarray:
    """
    Load frames from directory and return as tensor of type output_type.
    
    Args:
        frames_dir (str): Path to the directory containing the frames
        normalize (bool, optional): Whether to normalize pixel values to [0, 1]. 
                                   If False, returns values in [0, 255]. Default: False
        output_size (tuple[int, int], optional): Target size (height, width) to resize frames. Default: None
        output_type (Literal["pt", "np"], optional): Type of the output tensor. Default: "pt"
    Returns:
        torch.Tensor | np.ndarray: Frames tensor with shape (T, C, H, W) or (T, H, W, C) where T is number of frames,
                     C is number of channels, H is height, W is width
    """
    frames_dir = Path(frames_dir)
    file_list = sorted([p for p in frames_dir.rglob("*") if p.suffix.lower() in [".jpg", ".png"]])
    
    frames = []
    for frame_path in file_list:
        frame = imageio.imread(frame_path)
        frames.append(frame)
    frames = np.stack(frames) # (T, H, W, C)

    if output_size is not None:
        frames = _resize_frames(frames, output_size)

    if normalize:
        frames = frames.astype(np.float32) / 255.0

    if output_type == "np":
        return frames # (T, H, W, C)
    elif output_type == "pt":
        return torch.from_numpy(frames).permute(0, 3, 1, 2) # (T, C, H, W)
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def save_frames(
    frames: torch.Tensor | np.ndarray,
    save_dir: str,
    output_size: tuple[int, int] = None,
    format: Literal["png", "jpg"] = "png",
    pattern: str = "frame_{:06d}"
) -> None:
    """
    Save frames to directory.
    
    Args:
        frames (torch.Tensor | np.ndarray): Frames tensor with shape (T, C, H, W) or (T, H, W, C)
        save_dir (str): Path to the directory where to save the frames
        output_size (tuple[int, int], optional): Target size (height, width) to resize frames. Default: None
        format (Literal["png", "jpg"], optional): Output file format. Default: 'png'
        pattern (str, optional): Filename pattern with format placeholder for frame index. Default: 'frame_{:06d}'
    """
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8) # (T, H, W, C)
    elif isinstance(frames, np.ndarray):
        frames = frames.astype(np.uint8) # (T, H, W, C)
    else:
        raise ValueError(f"Invalid frames type: {type(frames)}")
    
    if output_size is not None:
        frames = _resize_frames(frames, output_size)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        filename = pattern.format(i) + f".{format}"
        imageio.imwrite(save_dir / filename, frame)
        