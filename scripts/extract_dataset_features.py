"""Extract model-selected attention features from the complete TAP-Vid DAVIS dataset."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from davis_profiles import DAVIS_PROFILES
from dotenv import load_dotenv

from heft import (
    ExtractionConfig,
    ExtractionTask,
    FeatureExtractionPool,
    TemporalMode,
)
from heft.attn_hook import FeatureKind
from heft.data import TapVidDavisDataset


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    model_name = os.environ["HEFT_MODEL"]
    temporal_mode = TemporalMode(os.getenv("HEFT_TEMPORAL_MODE", TemporalMode.POST_DIT))
    profile = DAVIS_PROFILES[model_name]
    model = profile.model
    model_path = os.getenv("HEFT_MODEL_PATH")
    if model_path:
        checkpoint = Path(model_path).expanduser().resolve()
        if not (checkpoint / "model_index.json").is_file():
            raise FileNotFoundError(f"no model_index.json in {checkpoint}")
        model = replace(model, model_id=str(checkpoint))
    dataset_root = Path(os.environ["HEFT_DATASET_ROOT"])
    model_cache = Path(os.environ["HEFT_MODEL_CACHE_DIR"])
    output_root = Path(
        os.environ["HEFT_FEATURE_OUTPUT_ROOT"].format(
            model=model_name, temporal_mode=temporal_mode.value
        )
    )
    gpu_ids = tuple(map(int, os.environ["HEFT_GPU_IDS"].split(",")))

    dataset = TapVidDavisDataset(dataset_root)
    samples = tuple(dataset[index] for index in range(len(dataset)))
    tasks = tuple(
        ExtractionTask(
            name=sample.video_id,
            input_video=sample.video.read(),
            output_dir=output_root / sample.video_id,
        )
        for sample in samples
    )
    capture = profile.capture
    if os.getenv("HEFT_CAPTURE_VALUE", "false").lower() == "true":
        capture = replace(capture, features=capture.features | {FeatureKind.VALUE})
    config = ExtractionConfig(
        capture=capture,
        chunk_size=25,
        max_pending_features=4,
        overwrite=os.getenv("HEFT_OVERWRITE", "false").lower() == "true",
        temporal_mode=temporal_mode,
    )
    with FeatureExtractionPool(
        model=model,
        config=config,
        gpu_ids=gpu_ids,
        cache_dir=model_cache,
    ) as extractor:
        results = extractor.extract(tasks)

    for result in results:
        print(
            result.task_name,
            [(chunk.chunk, chunk.gpu_id) for chunk in result.chunks],
            result.output_dir,
        )


if __name__ == "__main__":
    main()
