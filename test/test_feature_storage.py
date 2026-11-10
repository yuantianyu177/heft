"""Tests for grouped, random-access attention feature storage."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors import safe_open

import heft.attn_hook.storage as storage_module
from heft.attn_hook import (
    CapturedFeature,
    FeatureKind,
    FeatureRecorder,
    RecordedFeature,
    SafetensorsFeatureStorage,
)


def _feature(
    head: int,
    tensor: torch.Tensor,
    *,
    kind: FeatureKind = FeatureKind.QUERY,
    chunk: int = 1,
    step: int = 7,
    layer: int = 3,
) -> RecordedFeature:
    return RecordedFeature(
        kind=kind,
        chunk=chunk,
        step=step,
        layer=layer,
        head=head,
        tensor=tensor,
    )


def test_storage_groups_heads_and_loads_only_requested_heads(tmp_path: Path) -> None:
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[1, 3])
    path = tmp_path / "chunk_001" / "step_007" / "layer_003" / "query.safetensors"
    head_1 = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)
    head_3 = head_1 + 10

    storage(_feature(1, head_1))
    assert not path.exists()

    storage(_feature(3, head_3))
    assert path.is_file()
    assert "conditional" not in path.parts

    loaded = storage.load(
        chunk=1,
        step=7,
        layer=3,
        kind=FeatureKind.QUERY,
        heads=[3],
    )
    assert set(loaded) == {3}
    torch.testing.assert_close(loaded[3], head_3)

    with safe_open(path, framework="pt", device="cpu") as file:
        assert set(file.keys()) == {"head_001", "head_003"}
        assert file.metadata() == {
            "schema": "heft.attention_features",
            "schema_version": "1",
            "chunk": "1",
            "step": "7",
            "layer": "3",
            "kind": "query",
            "heads": "1,3",
        }

    storage.close()


def test_storage_separates_chunks_steps_layers_and_feature_kinds(
    tmp_path: Path,
) -> None:
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[0])

    storage(_feature(0, torch.tensor([1]), chunk=2, step=4, layer=6))
    storage(
        _feature(
            0,
            torch.tensor([2]),
            kind=FeatureKind.HIDDEN_STATES,
            chunk=2,
            step=4,
            layer=7,
        )
    )

    assert storage.path_for(chunk=2, step=4, layer=6, kind=FeatureKind.QUERY) == (
        tmp_path / "chunk_002" / "step_004" / "layer_006" / "query.safetensors"
    )
    assert storage.path_for(
        chunk=2, step=4, layer=7, kind=FeatureKind.HIDDEN_STATES
    ).is_file()
    storage.close()


def test_storage_rejects_unexpected_and_duplicate_heads(tmp_path: Path) -> None:
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[1, 3])

    with pytest.raises(ValueError, match="unexpected head 2"):
        storage(_feature(2, torch.tensor([2])))

    storage(_feature(1, torch.tensor([1])))
    with pytest.raises(ValueError, match="duplicate head 1"):
        storage(_feature(1, torch.tensor([1])))

    storage(_feature(3, torch.tensor([3])))
    with pytest.raises(ValueError, match="already been written"):
        storage(_feature(1, torch.tensor([1])))
    storage.close()


def test_storage_reports_incomplete_groups(tmp_path: Path) -> None:
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[1, 3])
    storage(_feature(1, torch.tensor([1])))

    with pytest.raises(RuntimeError, match=r"incomplete.*missing heads \[3\]"):
        storage.flush()
    with pytest.raises(RuntimeError, match=r"incomplete.*missing heads \[3\]"):
        storage.close()

    storage(_feature(3, torch.tensor([3])))
    storage.close()


def test_storage_refuses_to_overwrite_existing_files_by_default(tmp_path: Path) -> None:
    with SafetensorsFeatureStorage(tmp_path, expected_heads=[0]) as storage:
        storage(_feature(0, torch.tensor([1])))

    replacement = SafetensorsFeatureStorage(tmp_path, expected_heads=[0])
    with pytest.raises(FileExistsError, match="already exists"):
        replacement(_feature(0, torch.tensor([2])))

    with SafetensorsFeatureStorage(
        tmp_path, expected_heads=[0], overwrite=True
    ) as storage:
        storage(_feature(0, torch.tensor([3])))
    with SafetensorsFeatureStorage(tmp_path, expected_heads=[0]) as storage:
        loaded = storage.load(chunk=1, step=7, layer=3, kind=FeatureKind.QUERY)
    torch.testing.assert_close(loaded[0], torch.tensor([3]))


def test_storage_cleans_atomic_temporary_file_after_write_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_save(*_: object, **__: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(storage_module, "save_file", fail_save)
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[0])

    with pytest.raises(OSError, match="disk full"):
        storage(_feature(0, torch.tensor([1])))

    assert list(tmp_path.rglob("*.tmp")) == []


def test_storage_requires_owned_contiguous_cpu_tensors(tmp_path: Path) -> None:
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[0])
    non_contiguous = torch.arange(12).reshape(3, 4).transpose(0, 1)

    with pytest.raises(ValueError, match="contiguous"):
        storage(_feature(0, non_contiguous))

    if torch.cuda.is_available():
        with pytest.raises(ValueError, match="CPU"):
            storage(_feature(0, torch.tensor([1], device="cuda")))


def test_load_rejects_heads_missing_from_the_file(tmp_path: Path) -> None:
    with SafetensorsFeatureStorage(tmp_path, expected_heads=[1]) as storage:
        storage(_feature(1, torch.tensor([1])))

        with pytest.raises(KeyError, match="head 3"):
            storage.load(
                chunk=1,
                step=7,
                layer=3,
                kind=FeatureKind.QUERY,
                heads=[3],
            )


@pytest.mark.parametrize("kind", [FeatureKind.KEY, FeatureKind.VALUE])
def test_recorder_to_storage_round_trip(tmp_path: Path, kind: FeatureKind) -> None:
    storage = SafetensorsFeatureStorage(tmp_path, expected_heads=[0, 1])
    sources = {
        head: torch.arange(12).reshape(3, 4).transpose(0, 1) + head * 100
        for head in (0, 1)
    }

    with FeatureRecorder(sink=storage) as recorder:
        for head, tensor in sources.items():
            recorder(
                CapturedFeature(
                    kind=kind,
                    chunk=2,
                    step=5,
                    layer=7,
                    head=head,
                    tensor=tensor,
                )
            )
    storage.close()

    for head, expected in sources.items():
        actual = storage.load_head(
            chunk=2,
            step=5,
            layer=7,
            kind=kind,
            head=head,
        )
        assert actual.is_contiguous()
        torch.testing.assert_close(actual, expected)
