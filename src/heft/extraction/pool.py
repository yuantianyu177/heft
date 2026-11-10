"""Dynamic chunk routing across persistent per-GPU worker processes."""

from __future__ import annotations

import multiprocessing as mp
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Protocol, Self

import torch

from heft.tracking.features import FeatureVideo, FrameRange

from .config import (
    ChunkExecutionMetadata,
    ChunkJob,
    ChunkResult,
    ExtractionConfig,
    ExtractionResult,
    ExtractionTask,
    ModelConfig,
    TailFramePolicy,
    TemporalMode,
)
from .worker import DiffusersChunkExecutorFactory


class ChunkExecutor(Protocol):
    def __call__(self, job: ChunkJob) -> ChunkExecutionMetadata: ...


class ChunkExecutorFactory(Protocol):
    def __call__(self, gpu_id: int) -> ChunkExecutor: ...


_WORKER_GPU_ID: int | None = None
_WORKER_EXECUTOR: ChunkExecutor | None = None


def _initialize_worker(gpu_ids: Any, factory: ChunkExecutorFactory) -> None:
    global _WORKER_EXECUTOR, _WORKER_GPU_ID
    gpu_id: int = gpu_ids.get()
    _WORKER_GPU_ID = gpu_id
    _WORKER_EXECUTOR = factory(gpu_id)


def _execute_chunk(job: ChunkJob) -> ChunkResult:
    executor = _WORKER_EXECUTOR
    gpu_id = _WORKER_GPU_ID
    if executor is None or gpu_id is None:
        raise RuntimeError("feature extraction worker is not initialized")

    started_at = time.perf_counter()
    metadata = executor(job)
    return ChunkResult(
        task_index=job.task_index,
        task_name=job.task_name,
        output_dir=job.output_dir,
        chunk=job.chunk,
        start_frame=job.start_frame,
        end_frame=job.end_frame,
        seed=job.seed,
        gpu_id=gpu_id,
        elapsed_seconds=time.perf_counter() - started_at,
        heads=metadata.heads,
        feature_tokens=metadata.feature_tokens,
    )


class FeatureExtractionPool:
    """Route chunks to whichever selected GPU becomes available first."""

    def __init__(
        self,
        *,
        model: ModelConfig,
        config: ExtractionConfig,
        gpu_ids: Sequence[int],
        cache_dir: str | Path | None = None,
        executor_factory: ChunkExecutorFactory | None = None,
    ) -> None:
        normalized_gpu_ids = tuple(gpu_ids)
        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must not be empty")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must be unique")
        if not model.start_step <= config.capture.step < model.num_inference_steps:
            raise ValueError(
                "capture step must be inside the model's active denoising schedule"
            )

        self.model = model
        self.config = config
        self.gpu_ids = normalized_gpu_ids
        self._closed = False

        context = mp.get_context("spawn")
        gpu_queue = context.Queue()
        for gpu_id in normalized_gpu_ids:
            gpu_queue.put(gpu_id)
        factory = executor_factory or DiffusersChunkExecutorFactory(
            model=model,
            config=config,
            cache_dir=None if cache_dir is None else Path(cache_dir),
        )
        self._executor = ProcessPoolExecutor(
            max_workers=len(normalized_gpu_ids),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(gpu_queue, factory),
        )

    def map(self, tasks: Iterable[ExtractionTask]) -> tuple[ExtractionResult, ...]:
        """Extract all chunks and return results in task and frame order."""

        if self._closed:
            raise RuntimeError("feature extraction pool is closed")
        task_list = tuple(tasks)
        mode = TemporalMode(self.config.temporal_mode)
        if mode is not TemporalMode.POST_DIT and any(
            task.input_video is None for task in task_list
        ):
            raise ValueError(f"{mode.value} requires an input video")
        jobs = _plan_chunks(
            task_list,
            self.config.chunk_size,
            tail_policy=self.config.tail_policy,
        )
        futures = [self._executor.submit(_execute_chunk, job) for job in jobs]
        completed = [future.result() for future in as_completed(futures)]

        chunks_by_task: dict[int, list[ChunkResult]] = {
            index: [] for index in range(len(task_list))
        }
        for chunk in completed:
            chunks_by_task[chunk.task_index].append(chunk)

        results = tuple(
            ExtractionResult(
                task_name=task.name,
                output_dir=task.output_dir,
                chunks=tuple(
                    sorted(chunks_by_task[index], key=lambda chunk: chunk.chunk)
                ),
            )
            for index, task in enumerate(task_list)
        )
        for task, result in zip(task_list, results, strict=True):
            assert task.num_frames is not None
            heads = (
                tuple(self.config.capture.heads)
                if self.config.capture.heads
                else result.chunks[0].heads
            )
            feature_size = (
                self.model.resolution[0] // self.model.feature_spatial_stride[0],
                self.model.resolution[1] // self.model.feature_spatial_stride[1],
            )
            spatial_tokens = feature_size[0] * feature_size[1]
            feature_stride, first_feature_frame = self.model.feature_timing(
                input_video=task.input_video is not None,
                temporal_mode=mode,
            )
            FeatureVideo(
                root=result.output_dir,
                name=result.task_name,
                model=self.model.name,
                num_frames=result.chunks[-1].end_frame,
                frame_size=self.model.resolution,
                feature_size=feature_size,
                chunks=tuple(
                    FrameRange(
                        index=chunk.chunk,
                        start=chunk.start_frame,
                        stop=chunk.end_frame,
                        feature_frames=_feature_frames(chunk, spatial_tokens),
                        feature_frame_stride=feature_stride,
                        first_feature_frame=first_feature_frame,
                    )
                    for chunk in result.chunks
                ),
                step=self.config.capture.step,
                layers=tuple(self.config.capture.layers),
                heads=heads,
                feature_kinds=tuple(
                    feature.value for feature in self.config.capture.features
                ),
                rope=self.model.rope,
                temporal_mode=mode.value,
            ).write(overwrite=self.config.overwrite)
        return results

    def extract(self, tasks: Iterable[ExtractionTask]) -> tuple[ExtractionResult, ...]:
        """Extract a batch while keeping the GPU pipelines alive for reuse."""

        return self.map(tasks)

    def close(self) -> None:
        if self._closed:
            return
        self._executor.shutdown(wait=True)
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _plan_chunks(
    tasks: Sequence[ExtractionTask],
    chunk_size: int,
    *,
    tail_policy: TailFramePolicy | str = TailFramePolicy.PAD,
) -> tuple[ChunkJob, ...]:
    tail_policy = TailFramePolicy(tail_policy)
    jobs: list[ChunkJob] = []
    for task_index, task in enumerate(tasks):
        assert task.num_frames is not None
        stop_frame = task.num_frames
        if tail_policy is TailFramePolicy.DISCARD:
            stop_frame -= stop_frame % chunk_size
            if stop_frame == 0:
                raise ValueError(
                    f"{task.name}: discard policy leaves no complete chunk"
                )
        for chunk, start_frame in enumerate(range(0, stop_frame, chunk_size)):
            end_frame = min(start_frame + chunk_size, stop_frame)
            input_video = (
                None
                if task.input_video is None
                else task.input_video[start_frame:end_frame]
            )
            padding_frames = (
                chunk_size - (end_frame - start_frame)
                if tail_policy is TailFramePolicy.PAD
                else 0
            )
            if input_video is not None and padding_frames:
                padding = input_video[-1:].expand(padding_frames, -1, -1, -1)
                input_video = torch.cat((input_video, padding))
            jobs.append(
                ChunkJob(
                    task_index=task_index,
                    task_name=task.name,
                    output_dir=task.output_dir,
                    chunk=chunk,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    input_video=input_video,
                    prompt=task.prompt,
                    seed=task.seed,
                    padding_frames=padding_frames,
                )
            )
    return tuple(jobs)


FeatureExtractor = FeatureExtractionPool


def _feature_frames(chunk: ChunkResult, spatial_tokens: int) -> int:
    frames, remainder = divmod(chunk.feature_tokens, spatial_tokens)
    if remainder or frames == 0:
        raise ValueError(
            f"chunk {chunk.chunk} has {chunk.feature_tokens} tokens, which is not "
            f"divisible by its {spatial_tokens}-token spatial grid"
        )
    return frames
