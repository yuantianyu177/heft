"""Framewise video features with final-step inversion for the DiffTrack baseline."""

from __future__ import annotations

import torch
from torch import Tensor

from heft.attn_hook import AttentionProcessorKind, CaptureSpec, CapturedFeature, FeatureKind
from heft.extraction.config import ExtractionConfig, ModelConfig, TemporalMode
from heft.extraction.worker import DiffusersChunkExecutor, _prepare_input_video

from .difftrack import ddim_inverse_step, track_difftrack


class _Captured(Exception):
    """Stop the second transformer pass once the selected layer has been read."""


class DiffTrackExtractor:
    """Evaluate anchored Q/K matching with the frozen Wan or CogVideoX checkpoint."""

    @torch.inference_mode()
    def __init__(self, model: ModelConfig, layer: int, gpu: int, *, chunk_size: int = 13) -> None:
        if (model.processor not in (AttentionProcessorKind.WAN, AttentionProcessorKind.COGVIDEOX)
            or model.temporal_patch_size != 1 or model.start_step != model.num_inference_steps - 1):
            raise ValueError("DiffTrack needs the final step of a Wan or CogVideoX single-frame-patch model")
        self.model = model
        self.device = torch.device("cuda", gpu)
        self.chunk_size = chunk_size
        self.grid = tuple(n // s for n,s in zip(model.resolution, model.feature_spatial_stride, strict=True))
        self.wan = model.processor is AttentionProcessorKind.WAN
        spec = CaptureSpec(step=model.start_step, layers=(layer,), features=(FeatureKind.QUERY, FeatureKind.KEY))
        config = ExtractionConfig(capture=spec, chunk_size=chunk_size, temporal_mode=TemporalMode.FRAMEWISE)
        self.executor = DiffusersChunkExecutor(gpu_id=gpu, model=model, config=config, cache_dir=None)
        self.pipe = self.executor._pipeline
        if self.wan and (self.pipe.config.boundary_ratio is not None or self.pipe.config.expand_timesteps):
            raise ValueError("The flow inverse currently supports the Wan2.1 single-denoiser time input")
        self.scheduler = self.pipe.scheduler
        self.scheduler.set_timesteps(model.num_inference_steps, device=self.device)
        self.timestep = self.scheduler.timesteps[model.start_step]
        self.prompt, _ = self.pipe.encode_prompt(
            "", do_classifier_free_guidance=False, device=self.device,
            max_sequence_length=512 if self.wan else 226,
        )
        self.prompt = self.prompt.to(self.pipe.transformer.dtype)
        if self.wan:
            self.sigma = float(self.scheduler.sigmas[model.start_step])
            self.inverse_method = "explicit inverse flow step from sigma 0"
        else:
            self.alpha = float(self.scheduler.final_alpha_cumprod)
            self.target_alpha = float(self.scheduler.alphas_cumprod[int(self.timestep)])
            self.inverse_method = "one deterministic DDIM inverse step using native CogVideoX alphas"

    def _encode_frames(self, rgb: Tensor, batch_size: int = 4) -> Tensor:
        height, width = self.model.resolution
        encoded = []
        for batch in rgb.split(batch_size):
            prepared = _prepare_input_video(batch, self.model.resolution)
            clip = self.pipe.video_processor.preprocess_video(prepared, height=height, width=width)
            pixels = clip[0].permute(1, 0, 2, 3)[:, :, None].to(device=self.device, dtype=self.pipe.vae.dtype)
            latents = self.pipe.vae.encode(pixels).latent_dist.mode()
            if latents.shape[2] != 1:
                raise ValueError("Independent RGB frames must each produce a single latent frame")
            encoded.append(latents[:, :, 0])
        latents = torch.cat(encoded)
        if self.wan:
            mean = latents.new_tensor(self.pipe.vae.config.latents_mean).view(1, -1, 1, 1)
            std = latents.new_tensor(self.pipe.vae.config.latents_std).view(1, -1, 1, 1)
            return (latents - mean) / std
        return latents * self.pipe.vae_scaling_factor_image

    def _predict(self, sample: Tensor, timestep: Tensor, *, capture: bool = False) -> Tensor:
        transformer = self.pipe.transformer
        sample = sample.to(transformer.dtype)
        if self.wan:
            with transformer.cache_context("cond"):
                return transformer(
                    hidden_states=sample, timestep=timestep.expand(len(sample)),
                    encoder_hidden_states=self.prompt, return_dict=False,
                )[0]
        # The shared CogVideoX capture processor reads the conditional half.
        model_input = torch.cat((sample, sample)) if capture else sample
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        rotary = (
            self.pipe._prepare_rotary_positional_embeddings(*self.model.resolution, sample.shape[1], self.device)
            if transformer.config.use_rotary_positional_embeddings else None
        )
        with transformer.cache_context("cond_uncond" if capture else "cond"):
            return transformer(
                hidden_states=model_input, timestep=timestep.expand(len(model_input)),
                encoder_hidden_states=self.prompt.expand(len(model_input), -1, -1),
                image_rotary_emb=rotary, return_dict=False,
            )[0]

    def _chunk_qk(self, latents: Tensor, window: tuple[int, ...]) -> tuple[Tensor, Tensor]:
        frames = latents[list(window)].unsqueeze(0)
        if self.wan:
            clean = frames.transpose(1, 2).contiguous()
            velocity = self._predict(clean, torch.zeros_like(self.timestep))
            inverted = clean.float() + self.sigma * velocity.float()
        else:
            clean = frames
            prediction = self._predict(clean, self.timestep)
            inverted = ddim_inverse_step(clean.float(), prediction.float(), self.alpha, self.target_alpha,
                                         self.scheduler.config.prediction_type)

        captured: dict[tuple[FeatureKind, int], Tensor] = {}
        heads = self.executor._heads
        kinds = (FeatureKind.QUERY, FeatureKind.KEY)

        def collect(feature: CapturedFeature) -> None:
            captured[feature.kind, feature.head] = feature.tensor
            if len(captured) == len(heads) * len(kinds):
                raise _Captured

        self.executor._sink.bind(collect)
        try:
            self.executor._session.begin_chunk(chunk=0)
            self._predict(inverted, self.timestep, capture=True)
        except _Captured:
            pass
        finally:
            self.executor._session.end_chunk()
            self.executor._sink.unbind()

        spatial = self.grid[0] * self.grid[1]
        maps = []
        for kind in kinds:
            value = torch.cat([captured[kind, head] for head in heads], dim=-1)
            if value.shape[:2] != (1, len(window) * spatial):
                raise ValueError(f"Invalid {kind} token count {value.shape} for window {window}")
            maps.append(value.reshape(len(window), spatial, -1))
        return maps[0], maps[1]

    @torch.inference_mode()
    def track(self, rgb: Tensor, queries: Tensor) -> Tensor:
        if rgb.device.type != "cpu" or rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("Input RGB must be a CPU [time, 3, height, width] tensor")
        latents = self._encode_frames(rgb)
        return track_difftrack(
            len(rgb), queries.to(self.device), self.grid, lambda window: self._chunk_qk(latents, window),
            chunk_size=self.chunk_size,
        ).cpu()
