"""Select and emit per-head features exposed by attention processors."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from typing import Any, Protocol, Self

import torch
from torch import Tensor


class FeatureKind(StrEnum):
    """Features available at the attention processor capture point."""

    QUERY = "query"
    KEY = "key"
    VALUE = "value"
    HIDDEN_STATES = "hidden_states"
    ATTENTION_MAP = "attention_map"


class AttentionProcessorKind(StrEnum):
    """Model-specific processor invocation layouts supported by HeFT."""

    WAN = "wan"
    COSMOS = "cosmos"
    COSMOS3 = "cosmos3"
    COGVIDEOX = "cogvideox"


@dataclass(frozen=True, slots=True, init=False)
class CaptureSpec:
    """Immutable selection for one denoising step and one or more layers."""

    step: int
    layers: frozenset[int]
    heads: frozenset[int] | None
    features: frozenset[FeatureKind]

    def __init__(
        self,
        *,
        step: int,
        layers: Iterable[int],
        heads: Iterable[int] | None = None,
        features: Iterable[FeatureKind | str] = (
            FeatureKind.QUERY,
            FeatureKind.KEY,
            FeatureKind.HIDDEN_STATES,
            FeatureKind.ATTENTION_MAP,
        ),
    ) -> None:
        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("step must be an integer")
        if step < 0:
            raise ValueError("step must be non-negative")

        normalized_layers = _normalize_indices("layers", layers, allow_empty=False)
        normalized_heads = (
            None
            if heads is None
            else _normalize_indices("heads", heads, allow_empty=False)
        )
        normalized_features = frozenset(FeatureKind(feature) for feature in features)
        if not normalized_features:
            raise ValueError("features must not be empty")

        object.__setattr__(self, "step", step)
        object.__setattr__(self, "layers", normalized_layers)
        object.__setattr__(self, "heads", normalized_heads)
        object.__setattr__(self, "features", normalized_features)


def _normalize_indices(
    name: str, values: Iterable[int], *, allow_empty: bool
) -> frozenset[int]:
    normalized = frozenset(values)
    if not allow_empty and not normalized:
        raise ValueError(f"{name} must not be empty")
    if any(
        not isinstance(value, int) or isinstance(value, bool) for value in normalized
    ):
        raise TypeError(f"{name} must contain only integers")
    if any(value < 0 for value in normalized):
        raise ValueError(f"{name} must contain only non-negative integers")
    return normalized


@dataclass(frozen=True, slots=True)
class CaptureContext:
    """Default denoising metadata shared by processor capture sessions."""

    start_step: int = 0

    def __post_init__(self) -> None:
        _validate_non_negative_integer("start_step", self.start_step)


@dataclass(frozen=True, slots=True)
class CapturedFeature:
    """One feature kind for one denoising step, layer, and head."""

    kind: FeatureKind
    step: int
    layer: int
    head: int
    chunk: int
    tensor: Tensor


class FeatureSink(Protocol):
    """Synchronous consumer for a captured feature."""

    def __call__(self, feature: CapturedFeature) -> None: ...


class CaptureSession:
    """Manage one task-level attachment across multiple inference chunks."""

    def __init__(
        self,
        *,
        bindings: Sequence[tuple[Any, object]],
        callbacks: Sequence[_ProcessorCapture],
        default_start_step: int,
    ) -> None:
        self._bindings = list(bindings)
        self._callbacks = list(callbacks)
        self._default_start_step = default_start_step
        self._active_chunk: int | None = None
        self._removed = False

    def begin_chunk(self, *, chunk: int, start_step: int | None = None) -> None:
        """Enable capture and reset every selected layer for one chunk."""

        if self._removed:
            raise RuntimeError("capture session has been removed")
        if self._active_chunk is not None:
            raise RuntimeError(f"capture chunk {self._active_chunk} is already active")

        _validate_non_negative_integer("chunk", chunk)
        resolved_start_step = (
            self._default_start_step if start_step is None else start_step
        )
        _validate_non_negative_integer("start_step", resolved_start_step)

        for callback in self._callbacks:
            callback.begin_chunk(chunk=chunk, start_step=resolved_start_step)
        self._active_chunk = chunk

    def end_chunk(self) -> None:
        """Disable capture until the next chunk begins."""

        if self._active_chunk is None:
            return
        for callback in self._callbacks:
            callback.end_chunk()
        self._active_chunk = None

    def remove(self) -> None:
        self.end_chunk()
        while self._bindings:
            attention, previous = self._bindings.pop()
            if previous is _MISSING:
                delattr(attention, _CAPTURE_ATTRIBUTE)
            else:
                setattr(attention, _CAPTURE_ATTRIBUTE, previous)
        self._callbacks.clear()
        self._removed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.remove()


class AttentionFeatureCapture:
    """Capture conditional processor features and emit per-head records."""

    def __init__(
        self, *, spec: CaptureSpec, context: CaptureContext, sink: FeatureSink
    ) -> None:
        self.spec = spec
        self.context = context
        self.sink = sink

    def attach(
        self,
        blocks: Sequence[Any],
        *,
        processor: AttentionProcessorKind | str,
        attention_attribute: str = "attn1",
    ) -> CaptureSession:
        """Attach callbacks once and return their task-level session."""

        processor = AttentionProcessorKind(processor)

        attentions: list[tuple[int, Any]] = []
        for layer in sorted(self.spec.layers):
            if layer >= len(blocks):
                raise IndexError(
                    f"layer {layer} is outside the model's {len(blocks)} blocks"
                )
            block = blocks[layer]
            if not hasattr(block, attention_attribute):
                raise TypeError(
                    f"block at layer {layer} does not expose {attention_attribute}"
                )
            attentions.append((layer, getattr(block, attention_attribute)))

        bindings: list[tuple[Any, object]] = []
        callbacks: list[_ProcessorCapture] = []
        for layer, attention in attentions:
            previous = getattr(attention, _CAPTURE_ATTRIBUTE, _MISSING)
            bindings.append((attention, previous))
            target = partial(self._capture, layer=layer)
            callback: _ProcessorCapture
            if processor is AttentionProcessorKind.COGVIDEOX:
                callback = _BatchedConditionalCapture(target=target)
            else:
                callback = _AlternatingConditionalCapture(target=target)
            setattr(attention, _CAPTURE_ATTRIBUTE, callback)
            callbacks.append(callback)
        return CaptureSession(
            bindings=bindings,
            callbacks=callbacks,
            default_start_step=self.context.start_step,
        )

    def _capture(
        self,
        *,
        chunk: int,
        step: int,
        layer: int,
        query: Tensor,
        key: Tensor,
        hidden_states: Tensor,
        value: Tensor | None = None,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> None:
        """Capture canonical ``[batch, head, token, channel]`` tensors."""

        if step != self.spec.step or layer not in self.spec.layers:
            return

        _validate_feature_tensors(query, key, hidden_states)
        heads = self._selected_heads(query.shape[1])
        descriptors = {
            FeatureKind.QUERY: query,
            FeatureKind.KEY: key,
            FeatureKind.HIDDEN_STATES: hidden_states,
        }
        if FeatureKind.VALUE in self.spec.features:
            if value is None:
                raise ValueError(
                    "the attention processor did not provide a value tensor"
                )
            if value.ndim != 4 or value.shape[:3] != key.shape[:3]:
                raise ValueError(
                    "value must share key's batch, head, and token dimensions"
                )
            descriptors[FeatureKind.VALUE] = value

        for kind in FeatureKind:
            if kind not in self.spec.features:
                continue
            for head in heads:
                if kind is FeatureKind.ATTENTION_MAP:
                    tensor = _get_attention_map(
                        query=query[:, head].detach(),
                        key=key[:, head].detach(),
                        attention_mask=_attention_mask_for_head(
                            attention_mask, head, query.shape[1]
                        ),
                        is_causal=is_causal,
                    )
                else:
                    tensor = descriptors[kind][:, head].detach()
                self.sink(
                    CapturedFeature(
                        kind=kind,
                        step=step,
                        layer=layer,
                        head=head,
                        chunk=chunk,
                        tensor=tensor,
                    )
                )

    def _selected_heads(self, num_heads: int) -> tuple[int, ...]:
        heads = (
            tuple(range(num_heads))
            if self.spec.heads is None
            else tuple(sorted(self.spec.heads))
        )
        invalid_heads = [head for head in heads if head >= num_heads]
        if invalid_heads:
            raise IndexError(
                f"heads {invalid_heads} are outside the tensor's {num_heads} heads"
            )
        return heads


class _ProcessorCapture(Protocol):
    def begin_chunk(self, *, chunk: int, start_step: int) -> None: ...

    def end_chunk(self) -> None: ...

    def __call__(
        self,
        *,
        query: Tensor,
        key: Tensor,
        hidden_states: Tensor,
        value: Tensor | None = None,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> None: ...


class _CaptureTarget(Protocol):
    def __call__(
        self,
        *,
        chunk: int,
        step: int,
        query: Tensor,
        key: Tensor,
        hidden_states: Tensor,
        value: Tensor | None = None,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> None: ...


@dataclass(slots=True)
class _AlternatingConditionalCapture:
    """Count separate conditional/unconditional processor invocations."""

    target: _CaptureTarget
    _step: int = 0
    _is_conditional: bool = True
    _chunk: int | None = None

    def begin_chunk(self, *, chunk: int, start_step: int) -> None:
        self._chunk = chunk
        self._step = start_step
        self._is_conditional = True

    def end_chunk(self) -> None:
        self._chunk = None

    def __call__(
        self,
        *,
        query: Tensor,
        key: Tensor,
        hidden_states: Tensor,
        value: Tensor | None = None,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> None:
        chunk = self._chunk
        if chunk is None:
            return

        is_conditional = self._is_conditional
        step = self._step
        self._is_conditional = not self._is_conditional
        if not is_conditional:
            return

        self._step += 1
        self.target(
            chunk=chunk,
            step=step,
            query=query,
            key=key,
            hidden_states=hidden_states,
            value=value,
            attention_mask=attention_mask,
            is_causal=is_causal,
        )


@dataclass(slots=True)
class _BatchedConditionalCapture:
    """Count batched CFG calls and keep only their conditional half."""

    target: _CaptureTarget
    _step: int = 0
    _chunk: int | None = None

    def begin_chunk(self, *, chunk: int, start_step: int) -> None:
        self._chunk = chunk
        self._step = start_step

    def end_chunk(self) -> None:
        self._chunk = None

    def __call__(
        self,
        *,
        query: Tensor,
        key: Tensor,
        hidden_states: Tensor,
        value: Tensor | None = None,
        attention_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> None:
        chunk = self._chunk
        if chunk is None:
            return

        batch_size = query.shape[0]
        if batch_size < 2 or batch_size % 2 != 0:
            raise ValueError(
                "CogVideoX CFG capture requires a positive even batch size"
            )
        if key.shape[0] != batch_size or hidden_states.shape[0] != batch_size:
            raise ValueError(
                "query, key, and hidden_states must share the CFG batch size"
            )

        conditional_start = batch_size // 2
        step = self._step
        self._step += 1
        self.target(
            chunk=chunk,
            step=step,
            query=query[conditional_start:],
            key=key[conditional_start:],
            hidden_states=hidden_states[conditional_start:],
            value=value[conditional_start:] if value is not None else None,
            attention_mask=_conditional_attention_mask(
                attention_mask,
                batch_size=batch_size,
                conditional_start=conditional_start,
            ),
            is_causal=is_causal,
        )


def _conditional_attention_mask(
    attention_mask: Tensor | None,
    *,
    batch_size: int,
    conditional_start: int,
) -> Tensor | None:
    if (
        attention_mask is not None
        and attention_mask.ndim > 0
        and attention_mask.shape[0] == batch_size
    ):
        return attention_mask[conditional_start:]
    return attention_mask


def _validate_non_negative_integer(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")


def _validate_feature_tensors(
    query: Tensor, key: Tensor, hidden_states: Tensor
) -> None:
    for name, tensor in (
        ("query", query),
        ("key", key),
        ("hidden_states", hidden_states),
    ):
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [batch, head, token, channel]")
    if query.shape[:2] != key.shape[:2] or query.shape[:2] != hidden_states.shape[:2]:
        raise ValueError(
            "query, key, and hidden_states must share batch and head dimensions"
        )


def _attention_mask_for_head(
    attention_mask: Tensor | None,
    head: int,
    num_heads: int,
) -> Tensor | None:
    if (
        attention_mask is not None
        and attention_mask.ndim >= 4
        and attention_mask.shape[-3] == num_heads
    ):
        return attention_mask.select(-3, head)
    return attention_mask


@torch.no_grad()
def _get_attention_map(
    *,
    query: Tensor,
    key: Tensor,
    attention_mask: Tensor | None,
    is_causal: bool,
) -> Tensor:
    """Materialize one head's SDPA weights without allocating an identity value."""

    scores = torch.matmul(query, key.transpose(-2, -1))
    scores.mul_(query.shape[-1] ** -0.5)
    if attention_mask is not None:
        if attention_mask.dtype == torch.bool:
            scores.masked_fill_(~attention_mask, float("-inf"))
        else:
            scores.add_(attention_mask)
    if is_causal:
        query_length, key_length = scores.shape[-2:]
        causal_mask = torch.ones(
            (query_length, key_length),
            dtype=torch.bool,
            device=scores.device,
        ).tril()
        scores.masked_fill_(~causal_mask, float("-inf"))
    return scores.softmax(dim=-1)


_CAPTURE_ATTRIBUTE = "_heft_capture"
_MISSING = object()
