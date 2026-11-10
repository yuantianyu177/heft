from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from heft.attn_hook import FeatureKind
from heft.calibration import (
    AdaptiveLag,
    ContentLagEstimate,
    estimate_feature_video_lag,
    storage,
)
from heft.tracking import (
    FeatureSelection,
    FeatureVideo,
    FrameRange,
    RopePairing,
    RopeSpec,
)


def _captured_video(root: Path) -> FeatureVideo:
    chunks = (FrameRange(0, 0, 3, 2, 2), FrameRange(1, 3, 6, 3, 2, -1.0))
    video = FeatureVideo(
        root, "clock-test", "fake", 6, (8, 8), (2, 2), chunks,
        0, (1,), (2,), ("hidden_states",), RopeSpec(RopePairing.ADJACENT, 1, 1, 1),
    )
    video.write()
    for index, times in enumerate(((0., 4.), (10., 14., 18.))):
        group = root / f"chunk_{index:03d}" / "step_000" / "layer_001"
        group.mkdir(parents=True)
        values = torch.tensor(times).reshape(1, -1, 1, 1).expand(-1, -1, 4, 6).reshape(1, -1, 6)
        save_file({"head_002": values.contiguous()}, group / "hidden_states.safetensors")
    return video


def test_estimation_uses_nominal_tracker_readout_and_omits_duplicate_anchors(tmp_path, monkeypatch):
    video = _captured_video(tmp_path)
    images = torch.zeros(6, 3, 8, 8, dtype=torch.uint8)
    trace = ContentLagEstimate(0., False, 0, (), (), None, None)
    expected = AdaptiveLag(1.25, "prior", trace, None)

    def estimate(rgb, features, *, chunks, prior):
        assert rgb is images
        assert features.shape == (6, 6, 2, 2)
        assert chunks == ((0, 3), (3, 6))
        assert prior == 1.25
        torch.testing.assert_close(features[:, 0, 0, 0], torch.tensor([0., 2., 4., 12., 14., 16.]))
        return expected

    monkeypatch.setattr(storage, "estimate_video_lag", estimate)
    result = estimate_feature_video_lag(
        video.root, images, FeatureSelection(1, 2), kind=FeatureKind.HIDDEN_STATES, prior=1.25,
    )
    assert result == expected


def test_estimation_requires_rgb_for_each_feature_frame(tmp_path):
    video = _captured_video(tmp_path)
    with pytest.raises(ValueError, match="exactly the captured RGB frames"):
        estimate_feature_video_lag(video, torch.empty(5, 3, 8, 8), FeatureSelection(1, 2))
