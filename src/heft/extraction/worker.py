"""Persistent Diffusers executor owned by one GPU worker process."""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from heft.attn_hook import (
    AttentionFeatureCapture,
    AttentionProcessorKind,
    CaptureContext,
    CapturedFeature,
    FeatureKind,
    FeatureRecorder,
    FeatureSinkRouter,
    SafetensorsFeatureStorage,
)

from .config import (
    ChunkExecutionMetadata,
    ChunkJob,
    ExtractionConfig,
    ModelConfig,
    TemporalMode,
)
from .temporal import encode_video_latents, four_phases, stitch_phase_features


@dataclass(frozen=True, slots=True)
class DiffusersChunkExecutorFactory:
    """Picklable factory used to initialize one persistent GPU executor."""

    model: ModelConfig
    config: ExtractionConfig
    cache_dir: Path | None = None

    def __call__(self, gpu_id: int) -> DiffusersChunkExecutor:
        return DiffusersChunkExecutor(
            gpu_id=gpu_id,
            model=self.model,
            config=self.config,
            cache_dir=self.cache_dir,
        )


class DiffusersChunkExecutor:
    """Load a pipeline once and extract independently scheduled chunks."""

    def __init__(
        self,
        *,
        gpu_id: int,
        model: ModelConfig,
        config: ExtractionConfig,
        cache_dir: str | os.PathLike[str] | None,
    ) -> None:
        self._model = model
        self._config = config
        self._device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self._device)
        cv2.setNumThreads(1)

        module = importlib.import_module(model.pipeline_module)
        pipeline_class = getattr(module, model.pipeline_class)
        load_kwargs: dict[str, Any] = {"dtype": model.dtype}
        if cache_dir is not None:
            load_kwargs["cache_dir"] = str(cache_dir)
        if config.temporal_mode is TemporalMode.STRIDE1:
            load_kwargs["vae"] = _load_stride_one_vae(model, load_kwargs)
        self._pipeline = pipeline_class.from_pretrained(model.model_id, **load_kwargs)
        self._pipeline.to(self._device)

        transformer = self._pipeline.transformer
        self._blocks = getattr(transformer, model.blocks_attribute)
        first_layer = min(config.capture.layers)
        attention = getattr(self._blocks[first_layer], model.attention_attribute)
        num_heads = getattr(attention, "heads", None)
        if num_heads is None:
            num_heads = attention.num_attention_heads
        self._heads = (
            tuple(range(num_heads))
            if config.capture.heads is None
            else tuple(sorted(config.capture.heads))
        )

        self._sink = FeatureSinkRouter()
        capture = AttentionFeatureCapture(
            spec=config.capture,
            context=CaptureContext(start_step=model.start_step),
            sink=self._sink,
        )
        self._session = capture.attach(
            self._blocks,
            processor=model.processor,
            attention_attribute=model.attention_attribute,
        )

    @torch.inference_mode()
    def __call__(self, job: ChunkJob) -> ChunkExecutionMetadata:
        has_video = job.input_video is not None
        mode = TemporalMode(self._config.temporal_mode)
        if not has_video and mode is not TemporalMode.POST_DIT:
            raise ValueError(f"{mode.value} requires an input video")
        pipeline_frames, captured_frames = self._model.temporal_layout(
            job.pipeline_num_frames, input_video=has_video, temporal_mode=mode
        )
        _, retained_frames = self._model.temporal_layout(
            job.num_frames, input_video=has_video, temporal_mode=mode
        )
        height, width = self._model.resolution
        stride_h, stride_w = self._model.feature_spatial_stride
        spatial_tokens = (height // stride_h) * (width // stride_w)
        captured_tokens = captured_frames * spatial_tokens
        retained_tokens = retained_frames * spatial_tokens
        video = _prepare_input_video(job.input_video, (height, width))
        pixels = None
        if video is not None:
            pixels = self._pipeline.video_processor.preprocess_video(
                video, height=height, width=width
            ).to(device=self._device, dtype=self._pipeline.vae.dtype)

        storage = SafetensorsFeatureStorage(
            job.output_dir,
            expected_heads=self._heads,
            overwrite=self._config.overwrite,
        )
        with (
            storage,
            FeatureRecorder(
                sink=storage,
                max_pending=self._config.max_pending_features,
            ) as recorder,
        ):

            def sink(feature: CapturedFeature) -> None:
                if feature.tensor.shape[1] != captured_tokens:
                    raise ValueError(
                        f"{self._model.name} produced {feature.tensor.shape[1]} tokens "
                        f"for {pipeline_frames} frames, expected {captured_tokens}"
                    )
                if retained_tokens < captured_tokens:
                    feature = _trim_feature(feature, retained_tokens)
                recorder(feature)

            if mode is TemporalMode.FOUR_PHASE:
                assert pixels is not None
                self._capture_phases(job, pixels, spatial_tokens, recorder)
            else:
                latents = None
                if pixels is not None:
                    latents = encode_video_latents(
                        self._pipeline.vae,
                        pixels,
                        mode=mode,
                        native_stride=self._model.latent_frame_stride,
                        patch=self._model.temporal_patch_size,
                    )
                self._run_pipeline(
                    job,
                    job.pipeline_num_frames if has_video else pipeline_frames,
                    latents,
                    sink,
                )
        captured_kind = next(iter(self._config.capture.features))
        shape = recorder.feature_shape(captured_kind)
        if len(shape) != 3:
            raise ValueError(
                f"expected captured per-head features to be rank 3, got {shape}"
            )
        return ChunkExecutionMetadata(heads=self._heads, feature_tokens=shape[1])

    def _run_pipeline(
        self,
        job: ChunkJob,
        frames: int,
        latents: torch.Tensor | None,
        sink: Callable[[CapturedFeature], None],
    ) -> None:
        height, width = self._model.resolution
        start_step = self._model.start_step if latents is not None else 0
        inputs: dict[str, Any] = (
            {"input_latents": latents} if latents is not None else {"input_video": None}
        )
        if (
            latents is not None
            and self._model.processor is AttentionProcessorKind.COSMOS3
        ):
            native = self._config.temporal_mode in (
                TemporalMode.POST_DIT,
                TemporalMode.FOUR_PHASE,
            )
            inputs["input_latent_stride"] = (
                self._model.latent_frame_stride if native else 1
            )
        self._sink.bind(sink)
        try:
            self._session.begin_chunk(chunk=job.chunk, start_step=start_step)
            self._pipeline(
                prompt=job.prompt,
                height=height,
                width=width,
                num_frames=frames,
                num_inference_steps=self._model.num_inference_steps,
                guidance_scale=self._model.guidance_scale,
                generator=torch.Generator().manual_seed(job.seed),
                start_step=start_step,
                output_type="latent",
                **inputs,
            )
        finally:
            self._session.end_chunk()
            self._sink.unbind()

    def _capture_phases(
        self,
        job: ChunkJob,
        pixels: torch.Tensor,
        spatial_tokens: int,
        recorder: FeatureRecorder,
    ) -> None:
        stride = self._model.latent_frame_stride
        patch = self._model.temporal_patch_size
        phases = four_phases(pixels.shape[2], stride)
        records: list[dict[tuple[FeatureKind, int, int], CapturedFeature]] = []

        def collect(feature: CapturedFeature) -> None:
            key = feature.kind, feature.layer, feature.head
            records[-1][key] = replace(feature, tensor=feature.tensor.cpu())

        for phase in phases:
            records.append({})
            latents = encode_video_latents(
                self._pipeline.vae,
                pixels,
                mode=TemporalMode.FOUR_PHASE,
                native_stride=stride,
                patch=patch,
                phase=phase,
            )
            self._run_pipeline(job, job.pipeline_num_frames, latents, collect)
            if records[-1].keys() != records[0].keys():
                raise ValueError("captured descriptor groups differ between phases")

        for key, feature in records[0].items():
            tensor = stitch_phase_features(
                [record[key].tensor for record in records],
                phases,
                frames=job.num_frames,
                spatial_tokens=spatial_tokens,
                patch=patch,
            )
            recorder(replace(feature, tensor=tensor))


def _load_stride_one_vae(model: ModelConfig, load_kwargs: dict[str, Any]) -> Any:
    from diffusers import AutoencoderKLCogVideoX, AutoencoderKLWan

    if model.processor is AttentionProcessorKind.COGVIDEOX:
        return AutoencoderKLCogVideoX.from_pretrained(
            model.model_id, subfolder="vae", temporal_compression_ratio=1, **load_kwargs
        )
    return AutoencoderKLWan.from_pretrained(
        model.model_id,
        subfolder="vae",
        temperal_downsample=[False, False, False],
        scale_factor_temporal=1,
        **load_kwargs,
    )


def _prepare_input_video(
    video: torch.Tensor | None,
    size: tuple[int, int],
) -> torch.Tensor | None:
    if video is None:
        return video
    height, width = size
    frames = video.permute(0, 2, 3, 1).numpy()
    if tuple(video.shape[-2:]) != size:
        frames = np.stack(
            [
                cv2.resize(
                    frame,
                    (width, height),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                for frame in frames
            ]
        )
    prepared = torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
    if prepared.dtype is torch.uint8:
        prepared = prepared.to(dtype=torch.float32).div_(255.0)
    return prepared


def _trim_feature(feature: CapturedFeature, tokens: int) -> CapturedFeature:
    tensor = (
        feature.tensor[:, :tokens, :tokens]
        if feature.kind is FeatureKind.ATTENTION_MAP
        else feature.tensor[:, :tokens]
    )
    return replace(feature, tensor=tensor)
