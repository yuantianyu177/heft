"""Time expansion contracts for the six extraction backbones."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from heft import (
    COGVIDEOX,
    COGVIDEOX_1_5_5B,
    COSMOS_2,
    COSMOS_3_EDGE,
    WAN_2_1,
    WAN_2_2_TI2V_5B,
    ExtractionConfig,
    ModelConfig,
)
from heft.attn_hook import (
    AttentionFeatureCapture,
    AttentionProcessorKind,
    CaptureContext,
    CaptureSpec,
    FeatureKind,
    FeatureSinkRouter,
    SafetensorsFeatureStorage,
)
from heft.extraction import ChunkJob
from heft.extraction.worker import DiffusersChunkExecutor
from heft.tracking.ops import TrackingOps

MODELS = (
    WAN_2_1,
    WAN_2_2_TI2V_5B,
    COGVIDEOX,
    COGVIDEOX_1_5_5B,
    COSMOS_2,
    COSMOS_3_EDGE,
)
KINDS = (FeatureKind.QUERY, FeatureKind.KEY, FeatureKind.VALUE)
MODES = ("framewise", "stride1", "pre_dit", "post_dit", "four_phase")


class ToyVae:
    def __init__(self, stride: int) -> None:
        self.stride = stride
        self.dtype = torch.float32
        self.clips: list[Tensor] = []

    def encode(self, clip: Tensor) -> SimpleNamespace:
        pixels = clip[:, :1, :, :1, :1].clone()
        self.clips.append(pixels.flatten())
        return SimpleNamespace(
            latent_dist=SimpleNamespace(mode=lambda: pixels[:, :, :: self.stride])
        )


class ToyPipeline:
    def __init__(self, model: ModelConfig, stride: int) -> None:
        self.vae = ToyVae(stride)
        self.model = model
        self.video_processor = SimpleNamespace(
            preprocess_video=lambda video, **_: video.permute(1, 0, 2, 3).unsqueeze(0)
        )
        self.attention = SimpleNamespace()
        self.block = SimpleNamespace(**{model.attention_attribute: self.attention})
        self.latent_inputs: list[Tensor] = []

    def __call__(self, *, input_latents: Tensor, **_: object) -> None:
        raw = input_latents.flatten()
        self.latent_inputs.append(raw)
        pooled = raw.reshape(-1, self.model.temporal_patch_size).mean(dim=1)
        value = pooled.reshape(1, 1, -1, 1).expand(1, 1, -1, self.model.rope.channels)
        if self.model.processor is AttentionProcessorKind.COGVIDEOX:
            value = torch.cat((value - 100, value))
        self.attention._heft_capture(
            query=value + 1, key=value + 2, value=value, hidden_states=value
        )


@pytest.mark.parametrize("profile", MODELS, ids=lambda model: model.name)
@pytest.mark.parametrize("mode", MODES)
def test_temporal_mode_at_dit_and_on_disk(
    tmp_path: Path, profile: ModelConfig, mode: str
) -> None:
    model = replace(profile, resolution=profile.feature_spatial_stride)
    spec = CaptureSpec(step=model.start_step, layers=[0], heads=[0], features=KINDS)
    config = ExtractionConfig(capture=spec, chunk_size=9, temporal_mode=mode)
    pipe = ToyPipeline(model, stride=1 if mode == "stride1" else 4)
    sink = FeatureSinkRouter()
    capture = AttentionFeatureCapture(spec=spec, context=CaptureContext(), sink=sink)
    worker = DiffusersChunkExecutor.__new__(DiffusersChunkExecutor)
    worker._model, worker._config = model, config
    worker._pipeline, worker._device = pipe, torch.device("cpu")
    worker._heads, worker._sink = (0,), sink
    worker._session = capture.attach(
        [pipe.block],
        processor=model.processor,
        attention_attribute=model.attention_attribute,
    )
    pixels = torch.arange(9, dtype=torch.float32).square()
    video = pixels[:, None, None, None].expand(9, 3, *model.resolution).contiguous()
    result = worker(ChunkJob(0, "video", tmp_path, 0, 0, 9, video, "", 42))

    patch = model.temporal_patch_size
    dense_input_count = 10 if patch == 2 else 9
    native_input_count = 4 if patch == 2 else 3
    input_count = (
        native_input_count if mode in {"post_dit", "four_phase"} else dense_input_count
    )
    passes = 4 if mode == "four_phase" else 1
    assert [x.numel() for x in pipe.latent_inputs] == [input_count] * passes
    if mode == "framewise":
        assert [x.numel() for x in pipe.vae.clips] == [1] * 9
    else:
        assert [x.numel() for x in pipe.vae.clips] == [9] * passes
    if mode == "pre_dit":
        assert pipe.latent_inputs[0][:5].tolist() == [0, 4, 8, 12, 16]
    if mode == "four_phase":
        assert [x[1].item() for x in pipe.latent_inputs] == [16, 1, 4, 9]

    feature_times = 9 if mode == "four_phase" else input_count // patch
    assert result.feature_tokens == feature_times
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[0])
    tensors = [
        storage.load_head(chunk=0, step=spec.step, layer=0, kind=kind, head=0)
        for kind in KINDS
    ]
    assert all(
        tensor.shape == (1, feature_times, model.rope.channels) for tensor in tensors
    )
    torch.testing.assert_close(tensors[0], tensors[2] + 1)
    torch.testing.assert_close(tensors[1], tensors[2] + 2)
    if patch == 1 and mode in {"framewise", "stride1", "four_phase"}:
        torch.testing.assert_close(tensors[2][0, :, 0], pixels)
    if patch == 1 and mode == "post_dit":
        stride, offset = model.feature_timing(input_video=True, temporal_mode=mode)
        volume = tensors[2][0, :, :, None, None]
        readout = TrackingOps.interpolate_time(
            volume, 9, feature_frame_stride=stride, first_feature_frame=offset
        )
        assert readout[:, 0, 0, 0].tolist() == [0, 4, 8, 12, 16, 28, 40, 52, 64]


def test_four_phase_requires_descriptors_with_a_single_token_axis() -> None:
    spec = CaptureSpec(step=0, layers=[0], features=[FeatureKind.ATTENTION_MAP])
    with pytest.raises(ValueError, match="four_phase.*attention_map"):
        ExtractionConfig(capture=spec, chunk_size=9, temporal_mode="four_phase")
