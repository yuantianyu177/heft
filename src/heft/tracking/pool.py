"""Dynamic whole-video routing across persistent per-GPU tracking workers."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from typing import Any, Protocol, Self

import torch

from .config import TrackingConfig, TrackingResult, TrackingTask
from .tracker import FeatureTracker


class TrackingExecutor(Protocol):
    def __call__(self, task: TrackingTask) -> TrackingResult: ...


class TrackingExecutorFactory(Protocol):
    def __call__(self, gpu_id: int) -> TrackingExecutor: ...


@dataclass(frozen=True, slots=True)
class TorchTrackingExecutorFactory:
    config: TrackingConfig

    def __call__(self, gpu_id: int) -> TorchTrackingExecutor:
        return TorchTrackingExecutor(gpu_id=gpu_id, config=self.config)


class TorchTrackingExecutor:
    """Persistent tracker whose coordinate caches remain on one GPU."""

    def __init__(self, *, gpu_id: int, config: TrackingConfig) -> None:
        self.gpu_id = gpu_id
        self.device = torch.device("cuda", gpu_id)
        torch.cuda.set_device(self.device)
        self.tracker = FeatureTracker(config=config, device=self.device)

    def __call__(self, task: TrackingTask) -> TrackingResult:
        result = self.tracker.track(
            task.features,
            task.query_points,
            task.selection,
            video_id=task.video_id,
            time_advance_frames=task.time_advance_frames,
        )
        return replace(result, gpu_id=self.gpu_id)


@dataclass(frozen=True, slots=True)
class _IndexedTask:
    index: int
    task: TrackingTask


_WORKER_GPU_ID: int | None = None
_WORKER_EXECUTOR: TrackingExecutor | None = None


def _initialize_worker(gpu_ids: Any, factory: TrackingExecutorFactory) -> None:
    global _WORKER_EXECUTOR, _WORKER_GPU_ID
    gpu_id: int = gpu_ids.get()
    _WORKER_GPU_ID = gpu_id
    _WORKER_EXECUTOR = factory(gpu_id)


def _execute_tracking(job: _IndexedTask) -> tuple[int, TrackingResult]:
    executor = _WORKER_EXECUTOR
    if executor is None:
        raise RuntimeError("tracking worker is not initialized")
    return job.index, executor(job.task)


class TrackingPool:
    """Assign each complete video to whichever selected GPU becomes idle first."""

    def __init__(
        self,
        *,
        config: TrackingConfig | None = None,
        gpu_ids: Sequence[int],
        executor_factory: TrackingExecutorFactory | None = None,
    ) -> None:
        normalized_gpu_ids = tuple(gpu_ids)
        if not normalized_gpu_ids:
            raise ValueError("gpu_ids must not be empty")
        if len(set(normalized_gpu_ids)) != len(normalized_gpu_ids):
            raise ValueError("gpu_ids must be unique")
        self.config = TrackingConfig() if config is None else config
        self.gpu_ids = normalized_gpu_ids
        self._closed = False

        context = mp.get_context("spawn")
        gpu_queue = context.Queue()
        for gpu_id in normalized_gpu_ids:
            gpu_queue.put(gpu_id)
        factory = executor_factory or TorchTrackingExecutorFactory(self.config)
        self._executor = ProcessPoolExecutor(
            max_workers=len(normalized_gpu_ids),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(gpu_queue, factory),
        )

    def map(self, tasks: Iterable[TrackingTask]) -> tuple[TrackingResult, ...]:
        """Track tasks dynamically and restore their input ordering."""

        if self._closed:
            raise RuntimeError("tracking pool is closed")
        task_list = tuple(tasks)
        futures = [
            self._executor.submit(_execute_tracking, _IndexedTask(index, task))
            for index, task in enumerate(task_list)
        ]
        completed = [future.result() for future in as_completed(futures)]
        completed.sort(key=lambda item: item[0])
        return tuple(result for _, result in completed)

    def track(self, tasks: Iterable[TrackingTask]) -> tuple[TrackingResult, ...]:
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
