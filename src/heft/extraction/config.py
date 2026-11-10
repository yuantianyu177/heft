"""Configuration and task models for feature extraction."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import torch
from torch import Tensor

from heft.attn_hook import AttentionProcessorKind, CaptureSpec, FeatureKind
from heft.tracking.features import RopePairing, RopeSpec


class TailFramePolicy(StrEnum):
    """How to handle a final chunk shorter than ``chunk_size``."""

    PAD = "pad"
    DISCARD = "discard"


class TemporalMode(StrEnum):
    """How an input video is encoded and mapped to RGB-frame descriptors."""

    FRAMEWISE = "framewise"
    STRIDE1 = "stride1"
    PRE_DIT = "pre_dit"
    POST_DIT = "post_dit"
    FOUR_PHASE = "four_phase"


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Model-owned pipeline and inference parameters."""

    name: str
    model_id: str
    pipeline_class: str
    processor: AttentionProcessorKind
    blocks_attribute: str
    num_inference_steps: int
    start_step: int
    resolution: tuple[int, int]
    guidance_scale: float
    feature_spatial_stride: tuple[int, int]
    rope: RopeSpec
    pipeline_module: str = "diffusers"
    dtype: torch.dtype = torch.bfloat16
    attention_attribute: str = "attn1"
    latent_frame_stride: int = 1
    generated_latent_frame_stride: int | None = None
    temporal_patch_size: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "processor", AttentionProcessorKind(self.processor))
        if self.num_inference_steps <= 0:
            raise ValueError("num_inference_steps must be positive")
        if not 0 <= self.start_step < self.num_inference_steps:
            raise ValueError("start_step must be inside the denoising schedule")
        if self.guidance_scale <= 1:
            raise ValueError("guidance_scale must enable classifier-free guidance")
        if self.latent_frame_stride <= 0 or self.temporal_patch_size <= 0:
            raise ValueError(
                "latent frame stride and temporal patch size must be positive"
            )
        if (
            self.generated_latent_frame_stride is not None
            and self.generated_latent_frame_stride <= 0
        ):
            raise ValueError("generated latent frame stride must be positive")
        if (
            len(self.feature_spatial_stride) != 2
            or min(self.feature_spatial_stride) <= 0
        ):
            raise ValueError("feature_spatial_stride must contain positive values")
        if any(
            size % stride
            for size, stride in zip(
                self.resolution, self.feature_spatial_stride, strict=True
            )
        ):
            raise ValueError("resolution must be divisible by feature_spatial_stride")

    def _latent_stride(self, input_video: bool, mode: TemporalMode) -> int:
        if not input_video and self.generated_latent_frame_stride is not None:
            return self.generated_latent_frame_stride
        if input_video and mode is not TemporalMode.POST_DIT:
            return 1
        return self.latent_frame_stride

    def temporal_layout(
        self,
        frames: int,
        *,
        input_video: bool,
        temporal_mode: TemporalMode | str = TemporalMode.POST_DIT,
    ) -> tuple[int, int]:
        """Return padded RGB length and count of feature times covering the clip."""

        mode = TemporalMode(temporal_mode)
        if input_video and mode is TemporalMode.FOUR_PHASE:
            return frames, frames
        stride = self._latent_stride(input_video, mode)
        patch = self.temporal_patch_size
        latent_frames = (frames - 1 + stride - 1) // stride + 1
        feature_frames = (latent_frames + patch - 1) // patch
        return (feature_frames * patch - 1) * stride + 1, feature_frames

    def feature_timing(
        self,
        *,
        input_video: bool,
        temporal_mode: TemporalMode | str = TemporalMode.POST_DIT,
    ) -> tuple[int, float]:
        """Return the spacing and first center of the feature grid in RGB frames."""

        mode = TemporalMode(temporal_mode)
        if input_video and mode is TemporalMode.FOUR_PHASE:
            return 1, 0.0
        stride = self._latent_stride(input_video, mode)
        patch = self.temporal_patch_size
        return stride * patch, stride * (patch - 1) / 2


@dataclass(frozen=True, slots=True)
class ExtractionConfig:
    """Runtime settings shared by one homogeneous extraction pool."""

    capture: CaptureSpec
    chunk_size: int
    tail_policy: TailFramePolicy = TailFramePolicy.DISCARD
    max_pending_features: int = 8
    overwrite: bool = False
    temporal_mode: TemporalMode | str = TemporalMode.POST_DIT

    def __post_init__(self) -> None:
        object.__setattr__(self, "tail_policy", TailFramePolicy(self.tail_policy))
        mode = TemporalMode(self.temporal_mode)
        object.__setattr__(self, "temporal_mode", mode)
        if (
            mode is TemporalMode.FOUR_PHASE
            and FeatureKind.ATTENTION_MAP in self.capture.features
        ):
            raise ValueError(
                "four_phase supports token descriptors; attention_map spans independent contexts"
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.max_pending_features <= 0:
            raise ValueError("max_pending_features must be positive")


@dataclass(frozen=True, slots=True)
class ExtractionTask:
    """One input video whose chunks may run on different GPUs."""

    name: str
    output_dir: Path
    input_video: Tensor | None = None
    num_frames: int | None = None
    prompt: str = ""
    seed: int = 42

    def __init__(
        self,
        *,
        name: str,
        output_dir: str | os.PathLike[str],
        input_video: Tensor | None = None,
        num_frames: int | None = None,
        prompt: str = "",
        seed: int = 42,
    ) -> None:
        if input_video is None and num_frames is None:
            raise ValueError("num_frames is required without an input video")
        if input_video is not None:
            if input_video.ndim != 4:
                raise ValueError(
                    "input_video must have shape [frame, channel, height, width]"
                )
            if input_video.device.type != "cpu":
                raise ValueError("input_video must be a CPU tensor for multiprocessing")
            available_frames = input_video.shape[0]
            resolved_frames = available_frames if num_frames is None else num_frames
            if resolved_frames > available_frames:
                raise ValueError("num_frames exceeds the input video length")
        else:
            resolved_frames = num_frames
        if resolved_frames is None or resolved_frames <= 0:
            raise ValueError("num_frames must be positive")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "output_dir", Path(output_dir))
        object.__setattr__(self, "input_video", input_video)
        object.__setattr__(self, "num_frames", resolved_frames)
        object.__setattr__(self, "prompt", prompt)
        object.__setattr__(self, "seed", seed)


@dataclass(frozen=True, slots=True)
class ChunkJob:
    """One independently schedulable pipeline invocation."""

    task_index: int
    task_name: str
    output_dir: Path
    chunk: int
    start_frame: int
    end_frame: int
    input_video: Tensor | None
    prompt: str
    seed: int
    padding_frames: int = 0

    @property
    def num_frames(self) -> int:
        return self.end_frame - self.start_frame

    @property
    def pipeline_num_frames(self) -> int:
        return self.num_frames + self.padding_frames


@dataclass(frozen=True, slots=True)
class ChunkResult:
    """Execution metadata for one completed chunk."""

    task_index: int
    task_name: str
    output_dir: Path
    chunk: int
    start_frame: int
    end_frame: int
    seed: int
    gpu_id: int
    elapsed_seconds: float
    heads: tuple[int, ...]
    feature_tokens: int


@dataclass(frozen=True, slots=True)
class ChunkExecutionMetadata:
    """Shape metadata observed by a chunk executor while capturing features."""

    heads: tuple[int, ...]
    feature_tokens: int


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """All completed chunks for one input task, in frame order."""

    task_name: str
    output_dir: Path
    chunks: tuple[ChunkResult, ...]


WAN_2_1 = ModelConfig(
    name="wan2.1",
    model_id="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
    pipeline_class="WanPipeline",
    processor=AttentionProcessorKind.WAN,
    blocks_attribute="blocks",
    num_inference_steps=50,
    start_step=49,
    resolution=(480, 832),
    guidance_scale=5.0,
    feature_spatial_stride=(16, 16),
    latent_frame_stride=4,
    rope=RopeSpec(
        pairing=RopePairing.ADJACENT,
        temporal_pairs=22,
        height_pairs=21,
        width_pairs=21,
    ),
)

COSMOS_2 = ModelConfig(
    name="cosmos2",
    model_id="nvidia/Cosmos-Predict2-2B-Video2World",
    pipeline_class="Cosmos2VideoToWorldPipeline",
    processor=AttentionProcessorKind.COSMOS,
    blocks_attribute="transformer_blocks",
    num_inference_steps=35,
    start_step=34,
    resolution=(704, 1280),
    guidance_scale=7.0,
    feature_spatial_stride=(16, 16),
    latent_frame_stride=4,
    rope=RopeSpec(
        pairing=RopePairing.SPLIT_HALF,
        temporal_pairs=22,
        height_pairs=21,
        width_pairs=21,
    ),
)

COGVIDEOX = ModelConfig(
    name="cogvideox",
    model_id="THUDM/CogVideoX-2b",
    pipeline_class="CogVideoXPipeline",
    processor=AttentionProcessorKind.COGVIDEOX,
    blocks_attribute="transformer_blocks",
    num_inference_steps=50,
    start_step=49,
    resolution=(480, 720),
    guidance_scale=6.0,
    feature_spatial_stride=(16, 16),
    latent_frame_stride=4,
    rope=RopeSpec(
        pairing=RopePairing.ADJACENT,
        temporal_pairs=8,
        height_pairs=12,
        width_pairs=12,
    ),
)

WAN_2_2_TI2V_5B = ModelConfig(
    name="wan2.2-ti2v-5b",
    model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    pipeline_class="WanPipeline",
    processor=AttentionProcessorKind.WAN,
    blocks_attribute="blocks",
    num_inference_steps=50,
    start_step=49,
    resolution=(480, 832),
    guidance_scale=5.0,
    feature_spatial_stride=(32, 32),
    latent_frame_stride=4,
    rope=WAN_2_1.rope,
)

COGVIDEOX_1_5_5B = ModelConfig(
    name="cogvideox1.5-5b",
    model_id="zai-org/CogVideoX1.5-5B",
    pipeline_class="CogVideoXPipeline",
    processor=AttentionProcessorKind.COGVIDEOX,
    blocks_attribute="transformer_blocks",
    num_inference_steps=50,
    start_step=49,
    resolution=(480, 832),
    guidance_scale=6.0,
    feature_spatial_stride=(16, 16),
    latent_frame_stride=4,
    temporal_patch_size=2,
    rope=COGVIDEOX.rope,
)

COSMOS_3_EDGE = ModelConfig(
    name="cosmos3-edge",
    model_id="nvidia/Cosmos3-Edge",
    pipeline_class="Cosmos3OmniPipeline",
    processor=AttentionProcessorKind.COSMOS3,
    blocks_attribute="layers",
    attention_attribute="self_attn",
    num_inference_steps=50,
    start_step=49,
    resolution=(480, 832),
    guidance_scale=5.0,
    feature_spatial_stride=(32, 32),
    latent_frame_stride=4,
    rope=RopeSpec(
        pairing=RopePairing.INTERLEAVED_SPLIT_HALF,
        temporal_pairs=24,
        height_pairs=20,
        width_pairs=20,
    ),
)
