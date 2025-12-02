import torch
from torch import Tensor
from einops import rearrange
import torch.nn.functional as F
import numpy as np


def get_grid(height, width, patch_size=1):
    start_coord = patch_size // 2
    x = np.arange(start_coord, width, step=patch_size, dtype=np.float32)
    y = np.arange(start_coord, height, step=patch_size, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing='xy')
    grid = np.stack([X, Y], axis=-1)
    return torch.from_numpy(grid)

def sampler2d(input: Tensor, grid: Tensor, mode: str = 'bilinear'):
    '''
    Sample from a grid using bilinear interpolation.
    Args:
        input: (B, C, H, W)
        grid: (B, H, W, 2)
    Returns:
        output: (B, C, H, W)
    '''
    H, W = input.shape[-2:]
    xgrid, ygrid = grid.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    output = F.grid_sample(input, grid, align_corners=True, mode=mode)

    return output

def sampler3d(input: Tensor, points: Tensor, h: int, w: int, t: int):
    """
    Sample embeddings from an embeddings volume at specific points, using bilinear interpolation per timestep.

    Args:
        input (Tensor): a volume of embeddings/features. shape: (T, C, H, W)
        points (Tensor): batch of B points (pixel cooridnates) (x,y,t) you wish to sample. shape: (B, 3).
        h (int): Height of images.
        w (int): Width of images.
        t (int): number of frames.

    Returns:
        sampled_embeddings: sampled embeddings at specific positions. shape: (B, C).
    """
    input = rearrange(input, "t c h w -> 1 c t h w")
    samplers = points.detach().clone()
    if w > 1:
        samplers[:, 0] = (samplers[:, 0] / (w - 1)) * 2 - 1 # normalize to [-1,1]
    if h > 1:
        samplers[:, 1] = (samplers[:, 1] / (h - 1)) * 2 - 1 # normalize to [-1,1]
    if t > 1:
        samplers[:, 2] = (samplers[:, 2] / (t - 1)) * 2 - 1 # normalize to [-1,1]

    samples = rearrange(samplers, "b c -> 1 1 b 1 c")
    out = torch.nn.functional.grid_sample(input, samples, align_corners=True, padding_mode ='border')
    out = rearrange(out, "1 c 1 b 1 -> b c")
    return out

