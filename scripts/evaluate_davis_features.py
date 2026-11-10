"""Track and evaluate extracted video features on TAP-Vid DAVIS."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from math import isfinite
from pathlib import Path

from davis_profiles import DAVIS_PROFILES
from dotenv import load_dotenv

from heft import (
    AnnotationQueries,
    EvaluationConfig,
    FeatureVideo,
    TemporalMode,
    TrackingTask,
    evaluate_dataset,
    track_features,
)
from heft.calibration import estimate_feature_video_lag
from heft.data import TapVidDavisDataset


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model_name = os.environ["HEFT_MODEL"]
    temporal_mode = TemporalMode(os.getenv("HEFT_TEMPORAL_MODE", TemporalMode.POST_DIT))
    calibration = os.getenv("HEFT_TIME_CALIBRATION", "off")
    if calibration not in ("off", "adaptive"):
        raise ValueError("HEFT_TIME_CALIBRATION must be 'off' or 'adaptive'")
    prior = float(os.getenv("HEFT_TIME_PRIOR_FRAMES", "0")) if calibration == "adaptive" else 0.0
    if not isfinite(prior):
        raise ValueError("HEFT_TIME_PRIOR_FRAMES must be finite")
    profile = DAVIS_PROFILES[model_name]
    evaluation = EvaluationConfig()
    dataset = TapVidDavisDataset(os.environ["HEFT_DATASET_ROOT"])
    path_fields = {"model": model_name, "temporal_mode": temporal_mode.value, "time_calibration": calibration}
    feature_root = Path(os.environ["HEFT_FEATURE_OUTPUT_ROOT"].format(**path_fields))
    output_path = Path(os.environ["HEFT_EVALUATION_OUTPUT"].format(**path_fields))
    gpu_ids = tuple(map(int, os.environ["HEFT_GPU_IDS"].split(",")))
    samples = tuple(dataset[index] for index in range(len(dataset)))
    videos = {}
    for sample in samples:
        features = FeatureVideo.open(feature_root / sample.video_id)
        if features.num_frames != sample.video.num_frames:
            raise ValueError(
                f"{sample.video_id}: expected features for all {sample.video.num_frames} "
                f"video frames, got {features.num_frames}. "
                "Re-extract features into a fresh HEFT_FEATURE_OUTPUT_ROOT."
            )
        videos[sample.video_id] = features

    tasks = []
    clocks = {}
    for sample in samples:
        features = videos[sample.video_id]
        advance = 0.0
        if calibration == "adaptive":
            clock = estimate_feature_video_lag(
                features,
                sample.video.read(0, features.num_frames),
                profile.selection,
                kind=profile.tracking.query_feature,
                frequency_range=profile.tracking.frequency_range,
                prior=prior,
            )
            clocks[sample.video_id] = asdict(clock)
            advance = clock.advance_frames
        tasks.append(
            TrackingTask(
                features=features,
                query_points=AnnotationQueries(
                    sample.annotations, policy=evaluation.query_policy
                ).generate(features),
                selection=profile.selection,
                video_id=sample.video_id,
                time_advance_frames=advance,
            )
        )

    results = track_features(
        tasks,
        config=profile.tracking,
        gpu_ids=gpu_ids,
    )
    report = evaluate_dataset(results, dataset, config=evaluation)
    payload = asdict(report)
    if calibration == "adaptive":
        payload["time_calibration"] = {"mode": calibration, "prior_frames": prior, "videos": clocks}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
