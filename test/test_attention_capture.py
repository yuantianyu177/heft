"""Tests for selecting attention features before they leave the processor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
import torch

from heft.attn_hook import (
    AttentionFeatureCapture,
    AttentionProcessorKind,
    CaptureContext,
    CapturedFeature,
    CaptureSpec,
    FeatureKind,
)


@dataclass
class CollectingSink:
    records: list[CapturedFeature] = field(default_factory=list)

    def __call__(self, feature: CapturedFeature) -> None:
        self.records.append(feature)


class Attention:
    _heft_capture: Any


class Block:
    def __init__(self) -> None:
        self.attn1 = Attention()


def _invoke(callback: Any, tensor: torch.Tensor) -> None:
    callback(query=tensor, key=tensor, hidden_states=tensor)


def test_capture_spec_accepts_one_step_and_multiple_layers() -> None:
    spec = CaptureSpec(step=7, layers=[1, 3])

    assert spec.step == 7
    assert spec.layers == frozenset({1, 3})
    assert FeatureKind.VALUE not in spec.features

    with pytest.raises(TypeError, match="step must be an integer"):
        CaptureSpec(step=[7, 8], layers=[1, 3])  # type: ignore[arg-type]


def test_capture_selects_step_layer_head_and_all_feature_kinds() -> None:
    spec = CaptureSpec(
        step=7,
        layers=[1, 3],
        heads=[1],
        features=FeatureKind,
    )
    context = CaptureContext(start_step=6)
    sink = CollectingSink()
    capture = AttentionFeatureCapture(spec=spec, context=context, sink=sink)
    blocks = [Block() for _ in range(4)]
    session = capture.attach(blocks, processor=AttentionProcessorKind.WAN)

    query = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, -1.0]]]],
        requires_grad=True,
    )
    key = torch.tensor(
        [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]]],
        requires_grad=True,
    )
    value = torch.arange(8.0, 16.0).reshape(1, 2, 2, 2).requires_grad_()
    hidden_states = torch.arange(8.0).reshape(1, 2, 2, 2).requires_grad_()

    callback = blocks[1].attn1._heft_capture
    callback(query=query, key=key, value=value, hidden_states=hidden_states)
    assert sink.records == []

    session.begin_chunk(chunk=4)
    callback(query=query, key=key, value=value, hidden_states=hidden_states)
    assert sink.records == []

    # Wan executes conditional first and unconditional second at every step.
    callback(
        query=query + 100,
        key=key + 100,
        value=value + 100,
        hidden_states=hidden_states + 100,
    )
    assert sink.records == []

    callback(query=query, key=key, value=value, hidden_states=hidden_states)

    records = {record.kind: record for record in sink.records}
    assert set(records) == set(FeatureKind)
    assert all(record.step == 7 for record in records.values())
    assert all(record.layer == 1 for record in records.values())
    assert all(record.head == 1 for record in records.values())
    assert all(record.chunk == 4 for record in records.values())
    assert all(not record.tensor.requires_grad for record in records.values())
    torch.testing.assert_close(records[FeatureKind.QUERY].tensor, query[:, 1].detach())
    torch.testing.assert_close(records[FeatureKind.KEY].tensor, key[:, 1].detach())
    torch.testing.assert_close(records[FeatureKind.VALUE].tensor, value[:, 1].detach())
    torch.testing.assert_close(
        records[FeatureKind.HIDDEN_STATES].tensor, hidden_states[:, 1].detach()
    )

    expected_map = torch.softmax(
        query[:, 1] @ key[:, 1].transpose(-2, -1) / 2**0.5, dim=-1
    )
    torch.testing.assert_close(
        records[FeatureKind.ATTENTION_MAP].tensor, expected_map.detach()
    )


def test_capture_attaches_only_requested_layers_and_restores_previous_callback() -> (
    None
):
    blocks = [Block() for _ in range(4)]
    previous_capture = object()
    blocks[3].attn1._heft_capture = previous_capture
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=4, layers=[1, 3], features=[FeatureKind.QUERY]),
        context=CaptureContext(),
        sink=CollectingSink(),
    )

    session = capture.attach(blocks, processor=AttentionProcessorKind.COSMOS)

    assert not hasattr(blocks[0].attn1, "_heft_capture")
    assert callable(blocks[1].attn1._heft_capture)
    assert not hasattr(blocks[2].attn1, "_heft_capture")
    assert callable(blocks[3].attn1._heft_capture)

    session.remove()

    assert not hasattr(blocks[1].attn1, "_heft_capture")
    assert blocks[3].attn1._heft_capture is previous_capture


def test_query_only_capture_does_not_materialize_attention_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_matmul(*_: object) -> None:
        pytest.fail("query-only capture must not calculate an attention map")

    monkeypatch.setattr(torch, "matmul", unexpected_matmul)
    context = CaptureContext(start_step=2)
    sink = CollectingSink()
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=2, layers=[0], heads=[1], features=[FeatureKind.QUERY]),
        context=context,
        sink=sink,
    )
    tensor = torch.randn(1, 2, 4, 3)
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.WAN)
    session.begin_chunk(chunk=0)

    _invoke(blocks[0].attn1._heft_capture, tensor)

    assert [record.kind for record in sink.records] == [FeatureKind.QUERY]


def test_attention_maps_are_materialized_one_selected_head_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    matmul_shapes: list[torch.Size] = []
    original_matmul = torch.matmul

    def tracked_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        matmul_shapes.append(left.shape)
        return original_matmul(left, right)

    monkeypatch.setattr(torch, "matmul", tracked_matmul)
    context = CaptureContext(start_step=2)
    sink = CollectingSink()
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(
            step=2, layers=[0], heads=[0, 2], features=[FeatureKind.ATTENTION_MAP]
        ),
        context=context,
        sink=sink,
    )
    tensor = torch.randn(1, 3, 4, 3)
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.COSMOS)
    session.begin_chunk(chunk=0)

    _invoke(blocks[0].attn1._heft_capture, tensor)

    assert matmul_shapes == [torch.Size((1, 4, 3)), torch.Size((1, 4, 3))]
    assert [record.head for record in sink.records] == [0, 2]


def test_capture_submits_complete_feature_groups_in_storage_order() -> None:
    sink = CollectingSink()
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(
            step=0,
            layers=[0],
            heads=[0, 1],
            features=[FeatureKind.QUERY, FeatureKind.KEY],
        ),
        context=CaptureContext(),
        sink=sink,
    )
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.WAN)
    session.begin_chunk(chunk=0)

    _invoke(blocks[0].attn1._heft_capture, torch.randn(1, 2, 4, 3))

    assert [(record.kind, record.head) for record in sink.records] == [
        (FeatureKind.QUERY, 0),
        (FeatureKind.QUERY, 1),
        (FeatureKind.KEY, 0),
        (FeatureKind.KEY, 1),
    ]


def test_cogvideox_counts_each_call_and_keeps_second_cfg_batch_half() -> None:
    sink = CollectingSink()
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=6, layers=[0], heads=[1], features=[FeatureKind.QUERY]),
        context=CaptureContext(start_step=5),
        sink=sink,
    )
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.COGVIDEOX)
    session.begin_chunk(chunk=0)
    callback = blocks[0].attn1._heft_capture

    first_call = torch.arange(32.0).reshape(4, 2, 2, 2)
    second_call = first_call + 100
    _invoke(callback, first_call)
    assert sink.records == []

    _invoke(callback, second_call)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.step == 6
    torch.testing.assert_close(record.tensor, second_call[2:, 1])


def test_cogvideox_rejects_a_batch_that_cannot_be_split_for_cfg() -> None:
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=0, layers=[0], features=[FeatureKind.QUERY]),
        context=CaptureContext(),
        sink=CollectingSink(),
    )
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.COGVIDEOX)
    session.begin_chunk(chunk=0)

    with pytest.raises(ValueError, match="even batch size"):
        _invoke(blocks[0].attn1._heft_capture, torch.randn(3, 2, 4, 3))


def test_capture_session_resets_step_branch_and_chunk_without_reattaching() -> None:
    sink = CollectingSink()
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=7, layers=[0], heads=[0], features=[FeatureKind.QUERY]),
        context=CaptureContext(start_step=7),
        sink=sink,
    )
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.WAN)
    callback = blocks[0].attn1._heft_capture

    first = torch.randn(1, 1, 2, 2)
    second = torch.randn(1, 1, 2, 2)
    _invoke(callback, first)
    assert sink.records == []

    session.begin_chunk(chunk=2)
    _invoke(callback, first)
    session.end_chunk()
    _invoke(callback, first + 100)

    session.begin_chunk(chunk=3)
    assert blocks[0].attn1._heft_capture is callback
    _invoke(callback, second)
    session.end_chunk()

    assert [record.chunk for record in sink.records] == [2, 3]
    assert [record.step for record in sink.records] == [7, 7]
    torch.testing.assert_close(sink.records[0].tensor, first[:, 0])
    torch.testing.assert_close(sink.records[1].tensor, second[:, 0])


def test_capture_session_rejects_overlapping_chunks_and_use_after_remove() -> None:
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=0, layers=[0], features=[FeatureKind.QUERY]),
        context=CaptureContext(),
        sink=CollectingSink(),
    )
    blocks = [Block()]
    session = capture.attach(blocks, processor=AttentionProcessorKind.WAN)

    session.begin_chunk(chunk=0)
    with pytest.raises(RuntimeError, match="already active"):
        session.begin_chunk(chunk=1)

    session.end_chunk()
    session.end_chunk()
    session.remove()
    with pytest.raises(RuntimeError, match="removed"):
        session.begin_chunk(chunk=1)


def test_capture_session_restores_callbacks_after_an_exception() -> None:
    previous_capture = object()
    blocks = [Block()]
    blocks[0].attn1._heft_capture = previous_capture
    capture = AttentionFeatureCapture(
        spec=CaptureSpec(step=0, layers=[0], features=[FeatureKind.QUERY]),
        context=CaptureContext(),
        sink=CollectingSink(),
    )

    with (
        pytest.raises(RuntimeError, match="pipeline failed"),
        capture.attach(blocks, processor=AttentionProcessorKind.WAN) as session,
    ):
        session.begin_chunk(chunk=0)
        raise RuntimeError("pipeline failed")

    assert blocks[0].attn1._heft_capture is previous_capture
