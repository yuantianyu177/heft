import torch
from torch import Tensor
from pathlib import Path
from typing import Literal, List, Optional
from einops import rearrange
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import seaborn as sns
import math
import cv2
import multiprocessing as mp
from tqdm import tqdm
import os

from .utils.io import load_video

matplotlib.use("Agg")
plt.rcParams['font.family'] = 'serif'
# feature_kwargs: step, layer, head

class AttentionViewer:
    def __init__(
        self,
        data_dir: str,
        colormap: str = "mako",
        patch_size: int = 16,
        rope_dim: tuple[int, int, int] = (44, 42, 42), # (T, H, W)
        pooling_size: tuple[int, int] = (256, 256),
        device: str = "cuda:0",
    ):
        self.data_dir = Path(data_dir)
        self.colormap = colormap
        self.device = device
        self.rope_dim = rope_dim
        self.patch_size = patch_size
        self.video = load_video(self.data_dir / "video" / "original_video.mp4")
        self.num_frames = 25
        self.video_h, self.video_w = self.video.shape[-2:]
        self.token_h = self.video_h // patch_size
        self.token_w = self.video_w // patch_size
        self.num_tokens = self.token_h * self.token_w    
        self.pooling = nn.AdaptiveAvgPool2d(pooling_size)


    def get_token_path(self, **feature_kwargs) -> Path:
        step = feature_kwargs.get('step')
        layer = feature_kwargs.get('layer') 
        feature_type = feature_kwargs.get('feature_type')
        file_path = self.data_dir / feature_type / f"step_{step:03d}" / f"layer_{layer:02d}_chunk_000.pt"
        return file_path


    def get_token(self, **feature_kwargs) -> Tensor:
        file_path = self.get_token_path(**feature_kwargs)
        token = torch.load(file_path, map_location=self.device)
        head = feature_kwargs.get('head', -1)
        if head == -1:
            token = rearrange(token, "1 head n c -> n (head c)")
        else:
            token = rearrange(token, "1 head n c -> head n c")
            token = token[head]
        return token


    def extract_freq(self, token: Tensor, freq_range: tuple[float, float]=(0.0, 1.0)) -> Tensor:
        T, H, W = self.rope_dim
        token_t = token[:, int(T * freq_range[0]):int(T * freq_range[1])]
        token_h = token[:, T+int(H * freq_range[0]):T+int(H * freq_range[1])]
        token_w = token[:, T+H+int(42 * freq_range[0]):T+H+int(W * freq_range[1])]
        return torch.cat([token_t, token_h, token_w], dim=1)


    def get_attention(
        self, 
        query_type: Literal["query", "key", "hidden_states"], 
        key_type: Literal["query", "key", "hidden_states"],
        softmax: bool = False,
        pooling: bool = False,
        freq_range: tuple[float, float]=(0.0, 1.0),
        **feature_kwargs
    ) -> Tensor:
        query = self.get_token(**feature_kwargs, feature_type=query_type)
        key = self.get_token(**feature_kwargs, feature_type=key_type)
        query = self.extract_freq(query, freq_range) # (N, C)
        key = self.extract_freq(key, freq_range) 
        attention = (query @ key.transpose(-1, -2)) / math.sqrt(query.shape[-1]) # (N, N)
        if softmax:
            attention = F.softmax(attention, dim=-1)
        if pooling:
            attention = self.pooling(attention.unsqueeze(0).unsqueeze(0))
        return attention.squeeze(0).squeeze(0) # (N, N)


    def visualize_attention(
        self,
        query_type: Literal["query", "key", "hidden_states"] = "query",
        key_type: Literal["query", "key", "hidden_states"] = "key", 
        softmax: bool = True,
        pooling: bool = True,
        freq_range: tuple[float, float]=(0.0, 1.0),
        num_token: int = -1,
        title: str = None,
        output_path: str = "attention.png",
        **feature_kwargs
    ):  
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        attention = self.get_attention(query_type, key_type, softmax, pooling, freq_range, **feature_kwargs) # (N, N)
        attention = attention[:num_token, :num_token]
        if not isinstance(attention, torch.Tensor):
            attention = torch.tensor(attention)
        attention_4d = attention.unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        attention_ds = F.adaptive_avg_pool2d(attention_4d, (256, 256)).squeeze(0).squeeze(0)
        # plt.figure(figsize=(10, 8))
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(attention_ds.cpu().numpy(), cmap=self.colormap, cbar=False)
        # ax.set_title(title if title else f"{query_type}-{key_type} attention map (Freq Range {freq_range})", fontsize=26)
        ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        # cbar = ax.collections[0].colorbar
        # cbar.ax.tick_params(labelsize=16)
        plt.savefig(f"{output_path}", bbox_inches='tight')
        plt.close()


    def get_token_norm(
        self,
        feature_type: Literal["query", "key", "hidden_states"] = "query",
        **feature_kwargs
    ) -> Tensor:
        token = self.get_token(**feature_kwargs, feature_type=feature_type)
        T, H, W = self.rope_dim
        token_t = token[:, :T]
        token_h = token[:, T:T+H]
        token_w = token[:, T+H:T+H+W]
        
        def compute_pair_norms(token):
            token_pairs = token.unflatten(1, (-1, 2)) # (N, C_pairs, 2)
            norms = torch.norm(token_pairs, dim=-1)  # (N, C_pairs)
            return norms
        
        norms_t = compute_pair_norms(token_t)  # (N, T//2)
        norms_h = compute_pair_norms(token_h)  # (N, H//2) 
        norms_w = compute_pair_norms(token_w)  # (N, W//2)

        norms_t = rearrange(norms_t, "(t h w) c -> t h w c", t=self.num_frames, h=self.token_h)
        norms_h = rearrange(norms_h, "(t h w) c -> t h w c", t=self.num_frames, h=self.token_h)
        norms_w = rearrange(norms_w, "(t h w) c -> t h w c", t=self.num_frames, h=self.token_h)

        return norms_t[:, 0, 0, :], norms_h[0, :, 0, :], norms_w[0, 0, :, :]


    def visualize_token(
        self,
        norms: Tensor,
        num_token: int = -1,
        title: str = "frequency usage",
        output_path: str = "tmp",
        **feature_kwargs
    ):
        font_size = 24
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        norms = norms[:num_token].cpu().numpy().transpose(1, 0) # (C, N)
        # plt.figure(figsize=(10, 8))
        # ax = sns.heatmap(norms, cmap=self.colormap)
        plt.figure(figsize=(10, 10))
        ax = sns.heatmap(norms, cmap=self.colormap, cbar=False)
        ax.set_xticks([])
        ax.set_yticks([])
        # ax.set_title(title, fontsize=font_size)
        # num_rows = norms.shape[0]
        # ax.set_yticks([num_rows * 0.25, num_rows * 0.75])
        # ax.set_yticklabels(["High Frequencies", "Low Frequencies"], rotation=90, va="center", fontsize=font_size)
        # ax.tick_params(axis='y', length=0)
        # x_num = num_token if num_token != -1 else norms.shape[1]
        # x_ticks = np.arange(0, x_num, 4)
        # if x_ticks[-1] != x_num - 1:
        #     x_ticks = np.append(x_ticks, x_num - 1)
        # ax.set_xticks(x_ticks + 0.5)
        # ax.set_xticklabels(x_ticks)
        # ax.set_xlabel("Token", fontsize=font_size)
        # ax.tick_params(axis='x', labelsize=font_size-2)
        # cbar = ax.collections[0].colorbar
        # cbar.set_label("Norm", rotation=90, labelpad=18, fontsize=font_size)
        # cbar.ax.tick_params(labelsize=font_size-2)
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()


    def _process_single_patch_frame(
        self,
        args: tuple
    ) -> None:
        """
        Process a single query patch and key frame combination
        
        Args:
            args: Tuple containing (query_patch, patch_similarity, image, output_path)
        """
        (query_patch, patch_similarity, image, output_path) = args
        
        # Convert image to grayscale RGB
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.imshow(gray)
        
        # Add similarity patches
        for i, alpha in enumerate(patch_similarity):
            patch_h = i // self.token_w
            patch_w = i % self.token_w
            x1 = patch_w * self.patch_size
            y1 = patch_h * self.patch_size
            if alpha > 0.05:
                red_color = (1.0, 0.0, 0.0, alpha)
                rect = patches.Rectangle(
                    (x1, y1),
                    self.patch_size,
                    self.patch_size,
                    linewidth=0,
                    facecolor=red_color,
                    edgecolor='none'
                )
                ax.add_patch(rect)

        # Add query patch outline
        local_patch_id = query_patch % self.num_tokens
        query_patch_h = local_patch_id // self.token_w
        query_patch_w = local_patch_id % self.token_w
        query_x1 = query_patch_w * self.patch_size
        query_y1 = query_patch_h * self.patch_size
        query_rect = patches.Rectangle(
            (query_x1, query_y1),
            self.patch_size,
            self.patch_size,
            linewidth=3,
            facecolor='none',
            edgecolor='blue'
        )
        ax.add_patch(query_rect) 
        ax.axis('off')
        plt.tight_layout()
        
        # Save the figure
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

    def visualize_similarity(
        self,
        query_patches: List[int], 
        query_type: Literal["query", "key", "hidden_states"] = "query",
        key_type: Literal["query", "key", "hidden_states"] = "key", 
        output_dir: str = ".",
        **feature_kwargs
    ):
        similarity = self.get_attention(query_type, key_type, softmax=False, pooling=False, **feature_kwargs) # (N, N)
        similarity = similarity[query_patches] # (Q, N)
        similarity = rearrange(similarity, "q (t x) -> q t x", t=self.num_frames) # (Q, T, (H*W))
        similarity = F.softmax(similarity, dim=-1) # (Q, T, (H*W))
        max_similarity = similarity.max(dim=-1, keepdim=True)[0]
        min_similarity = similarity.min(dim=-1, keepdim=True)[0]
        similarity = (similarity - min_similarity) / (max_similarity - min_similarity + 1e-8) # (Q, T, (H*W))
        similarity = similarity.cpu().numpy() # (Q, T, (H*W))
        images = self.video.clone().cpu().numpy().transpose(0, 2, 3, 1) # (T, H, W, C)
        
        tasks = []
        for idx, query_patch in enumerate(query_patches):
            for key_frame in range(self.num_frames):
                patch_similarity = similarity[idx, key_frame] # (H*W,)
                image = images[key_frame] # (H, W, C)
                output_path = Path(output_dir) / f"{query_type}-{key_type}" / f"query_{query_patch}" \
                / f"step_{feature_kwargs['step']:03d}" / f"layer_{feature_kwargs['layer']:02d}" \
                / f"head_{feature_kwargs['head']:02d}" / f"frame_{key_frame:03d}.jpg"
                task_args = (query_patch, patch_similarity, image, output_path)
                tasks.append(task_args)

        # Execute tasks with multiprocessing and progress bar
        total_tasks = len(tasks)
        with mp.Pool(processes=min(total_tasks, os.cpu_count())) as pool:
            for _ in pool.imap(self._process_single_patch_frame, tasks): pass