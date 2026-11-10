import pytest
import torch
from safetensors.torch import save_file

from heft.attn_hook import FeatureKind
from heft.tracking import (
    FeatureSelection,
    FeatureTracker,
    FeatureVideo,
    FrameRange,
    RopePairing,
    RopeSpec,
    TrackingConfig,
    TrackingTask,
)
from heft.tracking.pool import TorchTrackingExecutor


def _delayed_capture(root):
    positions = ((1, 1, 2, 3), (4, 4, 5, 6))
    video = FeatureVideo(
        root=root,
        name="lagged",
        model="fake",
        num_frames=8,
        frame_size=(3, 8),
        feature_size=(3, 8),
        chunks=(FrameRange(0, 0, 4, 4, 1), FrameRange(1, 4, 8, 4, 1)),
        step=0,
        layers=(0,),
        heads=(0,),
        feature_kinds=("key",),
        rope=RopeSpec(RopePairing.ADJACENT, 1, 1, 1),
        temporal_mode="four_phase",
    )
    video.write()
    for i, x in enumerate(positions):
        volume = torch.zeros(4, 3, 8, 6)
        volume[..., 1] = 1
        volume[torch.arange(4), 1, torch.tensor(x)] = torch.tensor([1., 0., 0., 0., 0., 0.])
        target = root / f"chunk_{i:03d}" / "step_000" / "layer_000" / "key.safetensors"
        target.parent.mkdir(parents=True)
        save_file({"head_000": volume.reshape(1, -1, 6)}, target)
    return video


def test_task_advance_is_forwarded_and_applied_inside_each_clip(tmp_path):
    video = _delayed_capture(tmp_path)
    config = TrackingConfig(
        query_feature=FeatureKind.KEY,
        target_feature=FeatureKind.KEY,
        upsample_features=False,
        argmax_radius=0.0,
        search_radius=8.0,
        visibility_threshold=8.0,
    )
    executor = TorchTrackingExecutor.__new__(TorchTrackingExecutor)
    executor.tracker = FeatureTracker(config=config, device="cpu")
    executor.gpu_id = 0
    points = torch.tensor([[1., 1., 0.]])
    selection = FeatureSelection(0, 0)
    task_args = {"features": video.root, "query_points": points, "selection": selection}
    nominal = executor(TrackingTask(**task_args))
    corrected = executor(TrackingTask(**task_args, time_advance_frames=1.0))

    torch.testing.assert_close(nominal.tracks[0, :, 0], torch.tensor([1., 1., 2., 3., 4., 4., 5., 6.]))
    torch.testing.assert_close(corrected.tracks[0, :, 0], torch.tensor([1., 2., 3., 3., 4., 5., 6., 6.]))
    torch.testing.assert_close(corrected.tracks[0, :, 1], torch.ones(8))
    assert corrected.visibility.all()


def test_task_rejects_nonfinite_advance(tmp_path):
    with pytest.raises(ValueError, match="time_advance_frames must be finite"):
        TrackingTask(features=tmp_path, query_points=torch.empty(0, 3), selection=FeatureSelection(0, 0),
                     time_advance_frames=float("nan"))
