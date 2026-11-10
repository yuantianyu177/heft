"""Extract HEFT hidden descriptors from the exact Point Prompting backbone."""

from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor

from heft.extraction.temporal import (
    Phase,
    _sample_time,
    encode_video_latents,
    four_phases,
)

from .point_prompting_i2v import PointPromptingI2V


def stitch_reference_phases(
    captures: Sequence[Tensor],
    phases: Sequence[Phase],
    *,
    frames: int,
    spatial: int,
    patch: int,
) -> Tensor:
    """Interleave time patches using the native I2V front padding convention."""
    if not captures or len(captures) != len(phases):
        raise ValueError("Expected one capture per temporal phase")
    heads, _, channels = captures[0].shape
    dense = captures[0].new_empty(heads, frames, spatial, channels)
    anchors = []
    for capture, phase in zip(captures, phases, strict=True):
        real = phase.latent_times
        times = (0,) * ((-len(real)) % patch) + real
        centers = tuple(
            sum(times[i : i + patch]) / patch for i in range(0, len(times), patch)
        )
        output = capture.reshape(heads, len(centers), spatial, channels)
        kept = tuple(t for t in real if t < frames)
        volume = _sample_time(output, centers, kept, dim=1)
        anchors.append(volume[:, 0].float())
        dense[:, list(kept[1:])] = volume[:, 1:]
    dense[:, 0] = torch.stack(anchors).mean(dim=0).to(dense.dtype)
    return dense.flatten(1, 2)


class I2VHiddenExtractor:
    """Native first-image conditioning with four-phase feature capture."""

    @torch.inference_mode()
    def __init__(
        self,
        checkpoint: Path,
        family: str,
        gpu: int,
        *,
        resolution: tuple[int, int] = (480, 832),
        chunk_size: int = 25,
        text_encoder_source: Path | None = None,
    ):
        if chunk_size < 2:
            raise ValueError("The tracking chunk size must be at least two")
        self.engine = PointPromptingI2V(
            checkpoint,
            family,
            gpu,
            resolution=resolution,
            text_encoder_source=text_encoder_source,
        )
        self.wan, self.device = self.engine.wan, self.engine.device
        self.pipe = self.engine.pipe
        self.net = self.pipe.transformer
        self.vae = self.pipe.vae
        self.resolution, self.chunk_size = resolution, chunk_size
        self.blocks = self.net.blocks if self.wan else self.net.transformer_blocks
        self.num_heads = self.blocks[0].attn1.heads
        spatial = self.net.config.patch_size
        if self.wan:
            self.patch, self.patch_h, self.patch_w = spatial
        else:
            self.patch = self.net.config.patch_size_t or 1
            self.patch_h = self.patch_w = spatial
        spatial_stride = self.pipe.vae_scale_factor_spatial
        self.grid = (
            resolution[0] // (self.patch_h * spatial_stride),
            resolution[1] // (self.patch_w * spatial_stride),
        )
        self.pipe.scheduler.set_timesteps(50, device=self.device)
        self.timestep = self.pipe.scheduler.timesteps[-1]
        if hasattr(self.pipe.scheduler, "set_begin_index"):
            self.pipe.scheduler.set_begin_index(len(self.pipe.scheduler.timesteps) - 1)
        if self.wan:
            self.mean = torch.tensor(
                self.vae.config.latents_mean, device=self.device
            ).view(1, -1, 1, 1, 1)
            self.std = torch.tensor(
                self.vae.config.latents_std, device=self.device
            ).view(1, -1, 1, 1, 1)

    def _predict(self, clean, reference, frames):
        generator = torch.Generator(device=self.device).manual_seed(42)
        scheduler = self.pipe.scheduler
        timestep = self.timestep
        if self.wan:
            clean = (clean.float() - self.mean) / self.std
            noise = torch.randn(
                clean.shape, dtype=clean.dtype, device=self.device, generator=generator
            )
            sample = scheduler.add_noise(clean, noise, timestep[None])
            sample[:, :, 0] = reference[:, :, 0]
            labels = torch.ones(
                (1, clean.shape[2], *self.grid),
                device=self.device,
                dtype=timestep.dtype,
            )
            labels[:, 0] = 0
            labels = labels.flatten(1) * timestep
            kwargs = {}
        else:
            clean = (
                clean.permute(0, 2, 1, 3, 4).contiguous()
                * self.pipe.vae_scaling_factor_image
            )
            pad = (-clean.shape[1]) % self.patch
            clean = torch.cat((clean[:, :pad], clean), dim=1)
            condition = self.engine._condition_cog(reference, frames, clean, 42)
            noise = torch.randn(
                clean.shape, dtype=clean.dtype, device=self.device, generator=generator
            )
            sample = scheduler.add_noise(clean, noise, timestep[None])
            sample = scheduler.scale_model_input(sample, timestep).to(clean.dtype)
            sample = torch.cat((sample, condition), dim=2)
            labels = timestep[None]
            kwargs = {}
            if self.net.config.use_rotary_positional_embeddings:
                kwargs["image_rotary_emb"] = (
                    self.pipe._prepare_rotary_positional_embeddings(
                        *self.resolution, clean.shape[1], self.device
                    )
                )
            if self.net.config.ofs_embed_dim is not None:
                kwargs["ofs"] = clean.new_full((1,), 2.0)
        self.net(
            hidden_states=sample.to(self.net.dtype),
            timestep=labels,
            encoder_hidden_states=self.engine.prompt,
            return_dict=False,
            **kwargs,
        )

    def _capture_chunk(self, rgb: np.ndarray, layers: dict[int, tuple[int, ...]]):
        actual = len(rgb)
        padded = np.concatenate(
            (rgb, np.repeat(rgb[-1:], self.chunk_size - actual, axis=0))
        )
        pixels = self.engine._to_pixels(padded)
        phases = four_phases(self.chunk_size, 4)
        captured = {layer: [] for layer in layers}
        reference = pixels[:, :, :1]
        if self.wan:
            condition = self.vae.encode(reference).latent_dist.mode().float()
            reference = (condition - self.mean) / self.std
        attached = []

        def callback(layer, index):
            def collect(*, hidden_states, **_):
                captured[layer].append(hidden_states[0].index_select(0, index))

            return collect

        try:
            for layer, heads in layers.items():
                attention = self.blocks[layer].attn1
                old = getattr(attention, "_heft_capture", None)
                if old is not None:
                    raise RuntimeError(
                        "I2V capture requires an unoccupied attention hook"
                    )
                attached.append(attention)
                attention._heft_capture = callback(
                    layer, torch.tensor(heads, device=self.device)
                )
            for phase in phases:
                clean = encode_video_latents(
                    self.vae,
                    pixels,
                    mode="four_phase",
                    native_stride=4,
                    patch=1,
                    phase=phase,
                )
                self._predict(clean, reference, 1 + (len(phase.latent_times) - 1) * 4)
        finally:
            for attention in attached:
                delattr(attention, "_heft_capture")
        height, width = self.grid
        result = {}
        for layer, parts in captured.items():
            maps = stitch_reference_phases(
                parts, phases, frames=actual, spatial=height * width, patch=self.patch
            )
            result[layer] = (
                maps.reshape(len(layers[layer]), actual, height, width, -1)
                .permute(0, 1, 4, 2, 3)
                .contiguous()
            )
        return result

    @torch.inference_mode()
    def extract(
        self, rgb: np.ndarray, layers: dict[int, tuple[int, ...]]
    ) -> dict[int, Tensor]:
        """Return selected [head, time, channel, height, width] descriptors."""
        if rgb.ndim != 4 or rgb.dtype != np.uint8 or rgb.shape[-1] != 3 or not len(rgb):
            raise ValueError("Expected nonempty uint8 RGB [time, height, width, 3]")
        if not layers or any(
            not 0 <= layer < len(self.blocks)
            or not heads
            or min(heads) < 0
            or max(heads) >= self.num_heads
            for layer, heads in layers.items()
        ):
            raise ValueError("Invalid I2V attention sites")
        h, w = self.resolution
        prepared = np.stack(
            [cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR) for frame in rgb]
        )
        output = {layer: [] for layer in layers}
        for start in range(0, len(prepared), self.chunk_size):
            part = self._capture_chunk(
                prepared[start : start + self.chunk_size], layers
            )
            for layer, value in part.items():
                output[layer].append(value)
        return {layer: torch.cat(values, dim=1) for layer, values in output.items()}
