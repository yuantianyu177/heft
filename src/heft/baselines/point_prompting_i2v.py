"""Point Prompting with native first-frame conditioning on Wan and CogVideoX."""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from transformers import T5EncoderModel, UMT5EncoderModel

from diffusers import (
    CogVideoXDDIMScheduler,
    CogVideoXImageToVideoPipeline,
    WanImageToVideoPipeline,
)

from .point_prompting import (
    balance_red,
    counterfactual_denoise,
    marker_disk,
    trace_marker,
)


class PointPromptingI2V:
    """Regenerate the same clip twice, using a local mask in the second pass."""

    @torch.inference_mode()
    def __init__(
        self,
        checkpoint: Path,
        family: str,
        gpu: int,
        *,
        resolution: tuple[int, int],
        steps: int = 50,
        strength: float = 0.5,
        mask_radius: float = 20.0,
        window: int = 49,
        text_encoder_source: Path | None = None,
    ) -> None:
        if (
            family not in ("wan", "cog")
            or steps < 2
            or not 0 < strength <= 1
            or window < 2
        ):
            raise ValueError("Invalid I2V family or generation schedule")
        if int(steps * strength) < 1:
            raise ValueError(
                "The noise schedule must contain at least one denoising step"
            )
        self.wan = family == "wan"
        self.device = torch.device("cuda", gpu)
        torch.cuda.set_device(self.device)
        self.resolution = resolution
        self.steps, self.strength, self.mask_radius, self.window = (
            steps,
            strength,
            mask_radius,
            window,
        )
        pipeline = (
            WanImageToVideoPipeline if self.wan else CogVideoXImageToVideoPipeline
        )
        options = {}
        if text_encoder_source is not None:
            encoder = UMT5EncoderModel if self.wan else T5EncoderModel
            options["text_encoder"] = encoder.from_pretrained(
                str(text_encoder_source / "text_encoder"),
                dtype=torch.bfloat16,
                local_files_only=True,
            )
        self.pipe = pipeline.from_pretrained(
            str(checkpoint), dtype=torch.bfloat16, local_files_only=True, **options
        )
        if self.wan and (
            not self.pipe.config.expand_timesteps or self.pipe.transformer_2 is not None
        ):
            raise ValueError(
                "Expected a single-transformer Wan TI2V checkpoint with timestep masks"
            )
        if not self.wan and not isinstance(self.pipe.scheduler, CogVideoXDDIMScheduler):
            raise ValueError("Expected the native CogVideoX DDIM schedule")
        self.pipe.text_encoder.to(self.device)
        self.prompt = self.pipe.encode_prompt(
            prompt="",
            do_classifier_free_guidance=False,
            device=self.device,
        )[0].to(dtype=self.pipe.transformer.dtype)
        self.pipe.text_encoder = None
        self.pipe.transformer.to(self.device)
        self.pipe.vae.to(self.device)
        self.dtype = self.pipe.transformer.dtype
        self.stride = self.pipe.vae_scale_factor_temporal

    def _to_pixels(self, video: np.ndarray) -> Tensor:
        return (
            torch.as_tensor(video, device=self.device)
            .permute(3, 0, 1, 2)[None]
            .to(self.pipe.vae.dtype)
            / 127.5
            - 1.0
        )

    def _encode(self, pixels: Tensor) -> Tensor:
        latent = self.pipe.vae.encode(pixels).latent_dist.mode()
        if self.wan:
            return (latent.float() - self.mean) / self.std
        return (
            (latent * self.pipe.vae_scaling_factor_image)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )

    def _condition_cog(
        self, pixels: Tensor, frames: int, clean: Tensor, seed: int
    ) -> Tensor:
        height, width = self.resolution
        _, condition = self.pipe.prepare_latents(
            pixels[:, :, 0],
            1,
            clean.shape[2],
            frames,
            height,
            width,
            self.dtype,
            self.device,
            generator=torch.Generator(device=self.device).manual_seed(seed),
            latents=torch.zeros_like(clean),
        )
        return condition

    def _denoise(
        self,
        clean: Tensor,
        conditions: tuple[Tensor, Tensor],
        seed: int,
        *,
        extra_frames: int,
        mask: Tensor | None = None,
    ) -> Tensor:
        pipe = self.pipe
        scheduler = pipe.scheduler
        generator = torch.Generator(device=self.device).manual_seed(seed)
        scheduler.set_timesteps(self.steps, device=self.device)
        start = len(scheduler.timesteps) - int(self.steps * self.strength)
        times = scheduler.timesteps[start:]
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(start)
        noise = torch.randn(
            clean.shape, device=self.device, dtype=clean.dtype, generator=generator
        )
        if self.wan:
            patch_h, patch_w = pipe.transformer.config.patch_size[-2:]
            ref_mask = torch.ones_like(clean[:, :1])
            ref_mask[:, :, :1] = 0
        else:
            height, width = self.resolution
            rotary = pipe._prepare_rotary_positional_embeddings(
                height, width, clean.shape[1], self.device
            )
            ofs = (
                clean.new_full((1,), 2.0)
                if pipe.transformer.config.ofs_embed_dim is not None
                else None
            )

        def predict(sample, time, marked):
            condition = conditions[int(marked)]
            if self.wan:
                inputs = torch.where(ref_mask.bool(), sample, condition).to(self.dtype)
                step_labels = (ref_mask[:, 0, :, ::patch_h, ::patch_w] * time).flatten(
                    1
                )
                extras = {}
            else:
                model_input = scheduler.scale_model_input(sample, time).to(self.dtype)
                inputs = torch.cat((model_input, condition), dim=2)
                step_labels = time.expand(len(inputs))
                extras = {"image_rotary_emb": rotary, "ofs": ofs}
            return pipe.transformer(
                hidden_states=inputs,
                timestep=step_labels,
                encoder_hidden_states=self.prompt,
                return_dict=False,
                **extras,
            )[0]

        def step(sample, predicted, index):
            return scheduler.step(predicted, times[index], sample, return_dict=False)[
                0
            ].to(clean.dtype)

        latents = counterfactual_denoise(
            clean,
            noise,
            times,
            predict,
            step,
            lambda source, random, time: scheduler.add_noise(
                source, random, time[None]
            ),
            mask=mask,
        )
        if self.wan:
            latents[:, :, :1] = conditions[1]
            video = pipe.vae.decode(
                (latents * self.std + self.mean).to(pipe.vae.dtype)
            ).sample
        else:
            video = pipe.decode_latents(latents[:, extra_frames:].to(pipe.vae.dtype))
        rgb = (
            (video[0].permute(1, 2, 3, 0).float() * 127.5 + 127.5)
            .round()
            .clamp_(0, 255)
        )
        return rgb.to(torch.uint8).cpu().numpy()

    def _latent_mask(self, positions, padded_frames, extra_frames, size):
        points = np.concatenate(
            (
                positions,
                np.repeat(positions[-1:], padded_frames - len(positions), axis=0),
            )
        )
        disk = torch.from_numpy(
            marker_disk(points, self.resolution, self.mask_radius)
        ).to(self.device)
        pooled = F.adaptive_max_pool2d(disk[:, None].float(), size).bool()
        grouped = pooled[1:].reshape(-1, self.stride, *pooled.shape[1:]).any(dim=1)
        mask = torch.cat((pooled[:1], grouped))
        if self.wan:
            return mask.transpose(0, 1)[None]
        return torch.cat((mask[:extra_frames], mask))[None]

    def _track_window(self, video, marked_first, xy, seed):
        frames = len(video)
        pad = (-(frames - 1)) % self.stride
        video = np.concatenate((video, np.repeat(video[-1:], pad, axis=0)))
        pixels, marked = self._to_pixels(video), self._to_pixels(marked_first[None])
        extra = 0
        if self.wan:
            c = self.pipe.vae.config
            self.mean = torch.tensor(c.latents_mean, device=self.device).view(
                1, -1, 1, 1, 1
            )
            self.std = torch.tensor(c.latents_std, device=self.device).view(
                1, -1, 1, 1, 1
            )
            clean = self._encode(pixels)
            conditions = (self._encode(pixels[:, :, :1]), self._encode(marked))
        else:
            clean = self._encode(pixels)
            extra = (-clean.shape[1]) % (self.pipe.transformer.config.patch_size_t or 1)
            clean = torch.cat((clean[:, :extra], clean), dim=1)
            conditions = tuple(
                self._condition_cog(image, len(video), clean, seed)
                for image in (pixels, marked)
            )
        generated = self._denoise(clean, conditions, seed, extra_frames=extra)
        coarse, _ = trace_marker(generated[:frames], xy)
        mask = self._latent_mask(coarse, len(video), extra, clean.shape[-2:])
        generated = self._denoise(
            clean, conditions, seed, extra_frames=extra, mask=mask
        )
        return trace_marker(generated[:frames], xy)

    @torch.inference_mode()
    def track(self, rgb: np.ndarray, normalized_xy: np.ndarray, *, seed: int = 42):
        """Track a point through a complete video using overlapping generation windows."""
        height, width = self.resolution
        scale = np.array([width - 1, height - 1], dtype=np.float32)
        rgb = np.stack(
            [
                cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
                for frame in rgb
            ]
        )
        balanced = balance_red(rgb)
        positions = np.empty((len(rgb), 2), dtype=np.float32)
        visible = np.zeros(len(rgb), dtype=bool)
        positions[0] = np.clip(normalized_xy, 0.0, 1.0) * scale
        visible[0] = True
        for start in range(0, len(rgb) - 1, self.window - 1):
            stop = min(start + self.window, len(rgb))
            xy = positions[start]
            marked = balanced[start].copy()
            mark_size = (256, 256)
            mask = marker_disk((xy / scale)[None] * 255.0, mark_size, 2.0)[0].astype(
                np.uint8
            )
            resized = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            marked[resized.astype(bool)] = (255, 0, 0)
            p, v = self._track_window(
                balanced[start:stop], marked, tuple(xy), seed + start
            )
            positions[start + 1 : stop] = p[1:]
            visible[start + 1 : stop] = v[1:]
        return positions / scale, visible
