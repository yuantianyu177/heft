"""Tests for dynamically scheduling feature-extraction chunks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from heft import (
    COGVIDEOX,
    COGVIDEOX_1_5_5B,
    COSMOS_2,
    COSMOS_3_EDGE,
    WAN_2_1,
    WAN_2_2_TI2V_5B,
    ExtractionConfig,
    ExtractionTask,
    FeatureExtractionPool,
    FeatureExtractor,
    ModelConfig,
    TailFramePolicy,
)
from heft.attn_hook import (
    AttentionProcessorKind,
    CapturedFeature,
    CaptureSpec,
    FeatureKind,
)
from heft.extraction import ChunkExecutionMetadata, ChunkJob
from heft.extraction.pool import _plan_chunks
from heft.extraction.worker import _prepare_input_video, _trim_feature
from heft.tracking import FeatureVideo, RopePairing, RopeSpec


@dataclass(frozen=True, slots=True)
class FakeChunkExecutor:
    gpu_id: int
    delays: tuple[float, ...]

    def __call__(self, job: ChunkJob) -> ChunkExecutionMetadata:
        time.sleep(self.delays[job.chunk])
        return ChunkExecutionMetadata(
            heads=(0,),
            feature_tokens=job.num_frames * 30 * 52,
        )


@dataclass(frozen=True, slots=True)
class FakeChunkExecutorFactory:
    delays: tuple[float, ...]

    def __call__(self, gpu_id: int) -> FakeChunkExecutor:
        return FakeChunkExecutor(gpu_id=gpu_id, delays=self.delays)


def _model_config() -> ModelConfig:
    return ModelConfig(
        name="fake",
        model_id="unused",
        pipeline_class="UnusedPipeline",
        processor=AttentionProcessorKind.WAN,
        blocks_attribute="blocks",
        num_inference_steps=50,
        start_step=40,
        resolution=(480, 832),
        guidance_scale=5.0,
        feature_spatial_stride=(16, 16),
        rope=RopeSpec(
            pairing=RopePairing.ADJACENT,
            temporal_pairs=22,
            height_pairs=21,
            width_pairs=21,
        ),
    )


def _extraction_config(
    *,
    chunk_size: int,
    tail_policy: TailFramePolicy = TailFramePolicy.PAD,
) -> ExtractionConfig:
    return ExtractionConfig(
        capture=CaptureSpec(
            step=40,
            layers=[0],
            heads=[0],
            features=[FeatureKind.QUERY],
        ),
        chunk_size=chunk_size,
        tail_policy=tail_policy,
    )


def test_builtin_models_start_from_their_last_denoising_step() -> None:
    for model in (
        WAN_2_1,
        WAN_2_2_TI2V_5B,
        COSMOS_2,
        COSMOS_3_EDGE,
        COGVIDEOX,
        COGVIDEOX_1_5_5B,
    ):
        assert model.start_step == model.num_inference_steps - 1


def test_new_backbones_pad_the_codec_and_patch_grids() -> None:
    assert COGVIDEOX_1_5_5B.temporal_layout(25, input_video=True) == (29, 4)
    assert COGVIDEOX_1_5_5B.feature_timing(input_video=True) == (8, 2.0)
    assert COGVIDEOX_1_5_5B.temporal_layout(25, input_video=False) == (29, 4)
    for model in (WAN_2_1, WAN_2_2_TI2V_5B, COGVIDEOX, COSMOS_2, COSMOS_3_EDGE):
        assert model.temporal_layout(24, input_video=True) == (25, 7)
        assert model.feature_timing(input_video=True) == (4, 0.0)


def test_extraction_defaults_to_discarding_incomplete_tail() -> None:
    config = ExtractionConfig(
        capture=_extraction_config(chunk_size=5).capture,
        chunk_size=5,
    )

    assert config.tail_policy is TailFramePolicy.DISCARD


def test_uint8_dataset_video_is_normalized_for_diffusers() -> None:
    video = torch.tensor(
        [[[[0, 64], [128, 255]]] * 3],
        dtype=torch.uint8,
    )

    normalized = _prepare_input_video(video, (4, 4))

    assert normalized is not None
    assert normalized.dtype is torch.float32
    expected_frame = cv2.resize(
        video[0].permute(1, 2, 0).numpy(),
        (4, 4),
        interpolation=cv2.INTER_LANCZOS4,
    )
    expected = torch.from_numpy(np.asarray(expected_frame)).permute(2, 0, 1)[None]
    torch.testing.assert_close(normalized, expected.float() / 255.0)


def test_final_chunk_repeats_its_last_frame_for_pipeline_padding(
    tmp_path: Path,
) -> None:
    video = torch.arange(7).reshape(7, 1, 1, 1).expand(-1, 3, -1, -1)
    task = ExtractionTask(
        name="video-a",
        input_video=video,
        output_dir=tmp_path,
    )

    jobs = _plan_chunks(
        (task,),
        chunk_size=5,
        tail_policy=TailFramePolicy.PAD,
    )

    assert [(job.num_frames, job.pipeline_num_frames) for job in jobs] == [
        (5, 5),
        (2, 5),
    ]
    assert jobs[1].padding_frames == 3
    assert jobs[1].input_video is not None
    assert jobs[1].input_video[:, 0, 0, 0].tolist() == [5, 6, 6, 6, 6]


def test_padding_tokens_are_trimmed_before_recording() -> None:
    query = CapturedFeature(
        kind=FeatureKind.QUERY,
        step=1,
        layer=2,
        head=3,
        chunk=4,
        tensor=torch.zeros(1, 10, 6),
    )
    attention = CapturedFeature(
        kind=FeatureKind.ATTENTION_MAP,
        step=1,
        layer=2,
        head=3,
        chunk=4,
        tensor=torch.zeros(1, 10, 10),
    )

    assert _trim_feature(query, 7).tensor.shape == (1, 7, 6)
    assert _trim_feature(attention, 7).tensor.shape == (1, 7, 7)


def test_final_incomplete_chunk_can_be_discarded(tmp_path: Path) -> None:
    task = ExtractionTask(
        name="video-a",
        input_video=torch.zeros(7, 3, 1, 1),
        output_dir=tmp_path,
        seed=42,
    )

    jobs = _plan_chunks(
        (task,),
        chunk_size=5,
        tail_policy=TailFramePolicy.DISCARD,
    )

    assert [(job.start_frame, job.end_frame) for job in jobs] == [(0, 5)]
    assert jobs[0].padding_frames == 0
    assert jobs[0].seed == 42


def test_pool_splits_one_video_and_returns_chunks_in_frame_order(
    tmp_path: Path,
) -> None:
    video = torch.arange(5 * 3 * 2 * 2).reshape(5, 3, 2, 2)
    task = ExtractionTask(
        name="video-a",
        input_video=video,
        output_dir=tmp_path / "video-a",
        seed=100,
    )

    with FeatureExtractionPool(
        model=_model_config(),
        config=_extraction_config(chunk_size=2),
        gpu_ids=[3],
        executor_factory=FakeChunkExecutorFactory(delays=(0.0, 0.0, 0.0)),
    ) as pool:
        result = pool.map([task])[0]

    assert result.task_name == "video-a"
    assert [
        (chunk.chunk, chunk.start_frame, chunk.end_frame) for chunk in result.chunks
    ] == [
        (0, 0, 2),
        (1, 2, 4),
        (2, 4, 5),
    ]
    assert [chunk.seed for chunk in result.chunks] == [100, 100, 100]
    assert [chunk.gpu_id for chunk in result.chunks] == [3, 3, 3]

    feature_video = FeatureVideo.open(result.output_dir)
    assert feature_video.name == "video-a"
    assert feature_video.model == "fake"
    assert feature_video.num_frames == 5
    assert feature_video.frame_size == (480, 832)
    assert feature_video.feature_size == (30, 52)
    assert [(chunk.start, chunk.stop) for chunk in feature_video.chunks] == [
        (0, 2),
        (2, 4),
        (4, 5),
    ]
    assert [chunk.feature_frames for chunk in feature_video.chunks] == [2, 2, 1]
    assert feature_video.step == 40
    assert feature_video.layers == (0,)
    assert feature_video.heads == (0,)
    assert feature_video.feature_kinds == ("query",)
    assert feature_video.rope == _model_config().rope


def test_idle_gpu_dynamically_takes_more_chunks(tmp_path: Path) -> None:
    task = ExtractionTask(
        name="video-a",
        input_video=torch.zeros(5, 3, 2, 2),
        output_dir=tmp_path / "video-a",
    )

    with FeatureExtractionPool(
        model=_model_config(),
        config=_extraction_config(chunk_size=1),
        gpu_ids=[2, 6],
        executor_factory=FakeChunkExecutorFactory(
            delays=(0.30, 0.02, 0.02, 0.02, 0.02)
        ),
    ) as pool:
        result = pool.map([task])[0]

    chunks_per_gpu = {
        gpu_id: sum(chunk.gpu_id == gpu_id for chunk in result.chunks)
        for gpu_id in (2, 6)
    }
    assert sorted(chunks_per_gpu.values()) == [1, 4]
    assert [chunk.chunk for chunk in result.chunks] == list(range(5))


def test_pool_groups_interleaved_results_by_video(tmp_path: Path) -> None:
    tasks = [
        ExtractionTask(
            name=name,
            input_video=torch.zeros(frames, 3, 2, 2),
            output_dir=tmp_path / name,
            seed=seed,
        )
        for name, frames, seed in (("short", 2, 7), ("long", 5, 20))
    ]

    with FeatureExtractor(
        model=_model_config(),
        config=_extraction_config(chunk_size=2),
        gpu_ids=[0, 1],
        executor_factory=FakeChunkExecutorFactory(delays=(0.03, 0.01, 0.0)),
    ) as extractor:
        results = extractor.extract(tasks)

    assert [result.task_name for result in results] == ["short", "long"]
    assert [[chunk.chunk for chunk in result.chunks] for result in results] == [
        [0],
        [0, 1, 2],
    ]
    assert [[chunk.seed for chunk in result.chunks] for result in results] == [
        [7],
        [20, 20, 20],
    ]


def test_discard_policy_writes_retained_frame_count(tmp_path: Path) -> None:
    task = ExtractionTask(
        name="video-a",
        input_video=torch.zeros(7, 3, 2, 2),
        output_dir=tmp_path / "video-a",
    )

    with FeatureExtractionPool(
        model=_model_config(),
        config=_extraction_config(
            chunk_size=5,
            tail_policy=TailFramePolicy.DISCARD,
        ),
        gpu_ids=[0],
        executor_factory=FakeChunkExecutorFactory(delays=(0.0,)),
    ) as pool:
        result = pool.map([task])[0]

    assert [(chunk.start_frame, chunk.end_frame) for chunk in result.chunks] == [(0, 5)]
    assert FeatureVideo.open(result.output_dir).num_frames == 5
