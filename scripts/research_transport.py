"""Transport canonical key descriptors into the current spatial RoPE frame.

The model is untouched. Descriptor EMA occurs in the pre-RoPE coordinate
frame, and the query is rotated around the last predicted position when
matching. Invisible points use canonical global matching for re-detection.
"""
from __future__ import annotations

import torch
from torch import Tensor

from heft.tracking.ops import DescriptorState
from heft.tracking.pixel_ops import PixelTrackingOps

from research_rope import frequency_pairs


def transport_query(values: Tensor, positions: Tensor, *, frame_size, feature_size, cutoff=0.0):
    frame_h, frame_w = frame_size
    feature_h, feature_w = feature_size
    # align_corners=False token centers, consistent with spatial interpolation.
    coordinates = torch.stack(((positions[:, 1]+.5)*feature_h/frame_h-.5,
                               (positions[:, 0]+.5)*feature_w/frame_w-.5), dim=-1).double()
    frequencies = 10000.0 ** (-torch.arange(21, device=values.device, dtype=torch.float64) / 21)
    angles = torch.cat((torch.zeros(len(values), 22, device=values.device, dtype=torch.float64),
                        coordinates[:, 0, None]*frequencies, coordinates[:, 1, None]*frequencies), dim=-1)
    if cutoff:
        angles = angles[:, frequency_pairs(cutoff, 1).to(values.device)]
    cosine, sine = angles.cos().float(), angles.sin().float()
    u, v = values[:, 0::2], values[:, 1::2]
    return torch.stack((u*cosine-v*sine, v*cosine+u*sine), dim=-1).flatten(-2)


def make_transport_ops(canonical_volume, target_volume, *, feature_size, cutoff=0.0, state=None):
    """Create a scoped TrackingOps adapter with paired canonical/rotated maps."""
    if state is None:
        state = {"canonical_volume": canonical_volume, "target_volume": target_volume}

    class TransportOps(PixelTrackingOps):
        def frame(self, volume, index):
            frame = super().frame(volume, index)
            if volume.data_ptr() == state["target_volume"].data_ptr():
                self.canonical_target_frame = super().frame(state["canonical_volume"], index)
            return frame

        def predict(self, descriptors, target_frame, *, previous_positions, previous_visibility,
                    apply_search_mask, softmax_before_resize):
            output = torch.empty_like(previous_positions)
            visible = previous_visibility.nonzero().flatten()
            invisible = (~previous_visibility).nonzero().flatten()
            if visible.numel():
                transported = transport_query(
                    descriptors.values[visible], previous_positions[visible],
                    frame_size=(self.frame_h, self.frame_w), feature_size=feature_size, cutoff=cutoff,
                )
                output[visible] = super().predict(
                    DescriptorState.from_values(transported), target_frame,
                    previous_positions=previous_positions[visible], previous_visibility=previous_visibility[visible],
                    apply_search_mask=apply_search_mask, softmax_before_resize=softmax_before_resize,
                )
            if invisible.numel():
                output[invisible] = super().predict(
                    descriptors.index_select(invisible), self.canonical_target_frame,
                    previous_positions=previous_positions[invisible], previous_visibility=previous_visibility[invisible],
                    apply_search_mask=False, softmax_before_resize=softmax_before_resize,
                )
            return output

    return TransportOps
