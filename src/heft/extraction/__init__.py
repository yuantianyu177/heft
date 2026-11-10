"""Multi-GPU feature extraction API."""

from .api import extract_feature, extract_features
from .config import (
    COGVIDEOX,
    COGVIDEOX_1_5_5B,
    COSMOS_2,
    COSMOS_3_EDGE,
    WAN_2_1,
    WAN_2_2_TI2V_5B,
    ChunkExecutionMetadata,
    ChunkJob,
    ChunkResult,
    ExtractionConfig,
    ExtractionResult,
    ExtractionTask,
    ModelConfig,
    TailFramePolicy,
    TemporalMode,
)
from .pool import FeatureExtractionPool, FeatureExtractor

__all__ = [
    "COGVIDEOX",
    "COGVIDEOX_1_5_5B",
    "COSMOS_2",
    "COSMOS_3_EDGE",
    "WAN_2_1",
    "WAN_2_2_TI2V_5B",
    "ChunkExecutionMetadata",
    "ChunkJob",
    "ChunkResult",
    "ExtractionConfig",
    "ExtractionResult",
    "ExtractionTask",
    "FeatureExtractionPool",
    "FeatureExtractor",
    "ModelConfig",
    "TailFramePolicy",
    "TemporalMode",
    "extract_feature",
    "extract_features",
]
