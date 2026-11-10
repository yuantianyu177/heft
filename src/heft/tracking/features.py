"""Self-describing handles for extracted video features."""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Self

from heft.attn_hook import FeatureKind


class RopePairing(StrEnum):
    """Channel pairing used by a model's rotary position embedding."""

    ADJACENT = "adjacent"
    SPLIT_HALF = "split_half"
    INTERLEAVED_SPLIT_HALF = "interleaved_split_half"


@dataclass(frozen=True, slots=True)
class RopeSpec:
    """Model-specific RoPE channel layout, expressed in complex pairs."""

    pairing: RopePairing
    temporal_pairs: int
    height_pairs: int
    width_pairs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "pairing", RopePairing(self.pairing))
        pair_counts = (
            self.temporal_pairs,
            self.height_pairs,
            self.width_pairs,
        )
        if any(count <= 0 for count in pair_counts):
            raise ValueError("RoPE axis pair counts must be positive")
        if self.pairing is RopePairing.INTERLEAVED_SPLIT_HALF and (
            3 * self.height_pairs - 1 > self.num_pairs
            or 3 * self.width_pairs > self.num_pairs
        ):
            raise ValueError("interleaved RoPE axes exceed the head width")

    @property
    def num_pairs(self) -> int:
        return self.temporal_pairs + self.height_pairs + self.width_pairs

    @property
    def channels(self) -> int:
        return self.num_pairs * 2

    def channel_indices(self, frequency_range: tuple[float, float]) -> tuple[int, ...]:
        """Return both channels of every selected RoPE pair."""

        lower, upper = frequency_range
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("frequency_range must satisfy 0 <= lower < upper <= 1")
        if lower == 0.0 and upper == 1.0:
            return tuple(range(self.channels))

        if self.pairing is RopePairing.INTERLEAVED_SPLIT_HALF:
            height = range(1, 3 * self.height_pairs, 3)
            width = range(2, 3 * self.width_pairs, 3)
            temporal = tuple(
                pair
                for pair in range(self.num_pairs)
                if pair not in height and pair not in width
            )
            axes = (temporal, height, width)
        else:
            height_start = self.temporal_pairs
            width_start = height_start + self.height_pairs
            axes = (
                range(height_start),
                range(height_start, width_start),
                range(width_start, self.num_pairs),
            )
        selected_pairs = sorted(
            pair
            for axis in axes
            for pair in axis[int(len(axis) * lower) : int(len(axis) * upper)]
        )
        if not selected_pairs:
            raise ValueError("frequency_range selects no RoPE channels")

        if self.pairing is RopePairing.ADJACENT:
            return tuple(
                channel
                for pair in selected_pairs
                for channel in (pair * 2, pair * 2 + 1)
            )
        return tuple(
            selected_pairs + [pair + self.num_pairs for pair in selected_pairs]
        )


@dataclass(frozen=True, slots=True)
class FrameRange:
    """A source-frame range and the optional feature grid relative to its start."""

    index: int
    start: int
    stop: int
    feature_frames: int
    feature_frame_stride: int | None = None
    first_feature_frame: float = 0.0

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("chunk index must be non-negative")
        if self.start < 0 or self.stop <= self.start:
            raise ValueError("chunk frame range must be non-empty and non-negative")
        if self.feature_frames <= 0:
            raise ValueError("feature_frames must be positive")
        if self.feature_frame_stride is not None and self.feature_frame_stride <= 0:
            raise ValueError("feature_frame_stride must be positive")
        if not math.isfinite(self.first_feature_frame):
            raise ValueError("first_feature_frame must be finite")


@dataclass(frozen=True, slots=True)
class FeatureVideo:
    """Metadata required to track any previously extracted video."""

    root: Path
    name: str
    model: str
    num_frames: int
    frame_size: tuple[int, int]
    feature_size: tuple[int, int]
    chunks: tuple[FrameRange, ...]
    step: int
    layers: tuple[int, ...]
    heads: tuple[int, ...] | None
    feature_kinds: tuple[str, ...]
    rope: RopeSpec
    temporal_mode: str | None = None

    def __post_init__(self) -> None:
        root = Path(self.root)
        layers = tuple(sorted(set(self.layers)))
        heads = None if self.heads is None else tuple(sorted(set(self.heads)))
        feature_kinds = tuple(
            sorted({FeatureKind(kind).value for kind in self.feature_kinds})
        )
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "layers", layers)
        object.__setattr__(self, "heads", heads)
        object.__setattr__(self, "feature_kinds", feature_kinds)

        if not self.name:
            raise ValueError("feature video name must not be empty")
        if not self.model:
            raise ValueError("model name must not be empty")
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        _validate_size("frame_size", self.frame_size)
        _validate_size("feature_size", self.feature_size)
        if self.step < 0:
            raise ValueError("step must be non-negative")
        if not layers or any(layer < 0 for layer in layers):
            raise ValueError("layers must contain non-negative indices")
        if heads is not None and (not heads or any(head < 0 for head in heads)):
            raise ValueError("heads must be None or contain non-negative indices")
        if not feature_kinds:
            raise ValueError("feature_kinds must not be empty")
        _validate_chunks(self.chunks, self.num_frames)

    @property
    def manifest_path(self) -> Path:
        return self.root / _MANIFEST_NAME

    def write(self, *, overwrite: bool = False) -> Path:
        """Atomically persist this feature video's manifest."""

        path = self.manifest_path
        if path.exists() and not overwrite:
            raise FileExistsError(f"feature manifest already exists: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": _SCHEMA,
            "schema_version": _SCHEMA_VERSION,
            "name": self.name,
            "model": self.model,
            "num_frames": self.num_frames,
            "frame_size": list(self.frame_size),
            "feature_size": list(self.feature_size),
            "temporal_mode": self.temporal_mode,
            "chunks": [
                {
                    "index": chunk.index,
                    "start": chunk.start,
                    "stop": chunk.stop,
                    "feature_frames": chunk.feature_frames,
                    "feature_frame_stride": chunk.feature_frame_stride,
                    "first_feature_frame": chunk.first_feature_frame,
                }
                for chunk in self.chunks
            ],
            "capture": {
                "step": self.step,
                "layers": list(self.layers),
                "heads": None if self.heads is None else list(self.heads),
                "feature_kinds": list(self.feature_kinds),
                "rope": {
                    "pairing": self.rope.pairing.value,
                    "temporal_pairs": self.rope.temporal_pairs,
                    "height_pairs": self.rope.height_pairs,
                    "width_pairs": self.rope.width_pairs,
                },
            },
        }
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, sort_keys=True)
                file.write("\n")
            os.replace(temporary_path, path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
        return path

    @classmethod
    def open(cls, root: str | os.PathLike[str]) -> Self:
        """Load and validate a feature-video manifest."""

        resolved_root = Path(root)
        path = resolved_root / _MANIFEST_NAME
        with path.open(encoding="utf-8") as file:
            payload: dict[str, Any] = json.load(file)
        if (
            payload.get("schema") != _SCHEMA
            or payload.get("schema_version") != _SCHEMA_VERSION
        ):
            raise ValueError(f"unsupported feature manifest: {path}")
        capture = payload["capture"]
        chunks = tuple(FrameRange(**chunk) for chunk in payload["chunks"])
        raw_heads = capture["heads"]
        return cls(
            root=resolved_root,
            name=payload["name"],
            model=payload["model"],
            num_frames=payload["num_frames"],
            frame_size=tuple(payload["frame_size"]),
            feature_size=tuple(payload["feature_size"]),
            chunks=chunks,
            step=capture["step"],
            layers=tuple(capture["layers"]),
            heads=None if raw_heads is None else tuple(raw_heads),
            feature_kinds=tuple(capture["feature_kinds"]),
            rope=RopeSpec(**capture["rope"]),
            temporal_mode=payload.get("temporal_mode"),
        )


def _validate_size(name: str, size: tuple[int, int]) -> None:
    if len(size) != 2 or min(size) <= 0:
        raise ValueError(f"{name} must contain positive height and width")


def _validate_chunks(chunks: tuple[FrameRange, ...], num_frames: int) -> None:
    if not chunks:
        raise ValueError("chunks must not be empty")
    expected_start = 0
    for expected_index, chunk in enumerate(chunks):
        if chunk.index != expected_index:
            raise ValueError("chunk indices must be contiguous and start at zero")
        if chunk.start != expected_start:
            raise ValueError("chunk frame ranges must be contiguous")
        expected_start = chunk.stop
    if expected_start != num_frames:
        raise ValueError("chunk frame ranges must cover the complete video")


_MANIFEST_NAME = "manifest.json"
_SCHEMA = "heft.feature_video"
_SCHEMA_VERSION = 2
