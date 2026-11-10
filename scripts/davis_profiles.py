"""Model profiles shared by DAVIS extraction and evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass

from heft import (
    COGVIDEOX,
    COGVIDEOX_1_5_5B,
    COSMOS_2,
    COSMOS_3_EDGE,
    WAN_2_1,
    WAN_2_2_TI2V_5B,
    ModelConfig,
    TrackingConfig,
)
from heft.attn_hook import CaptureSpec, FeatureKind
from heft.tracking import FeatureSelection


@dataclass(frozen=True, slots=True)
class DavisProfile:
    model: ModelConfig
    layer: int
    head: int
    tracking: TrackingConfig

    @property
    def capture(self) -> CaptureSpec:
        return CaptureSpec(
            step=self.model.start_step,
            layers=(self.layer,),
            heads=(self.head,),
            features=(FeatureKind.QUERY, FeatureKind.KEY),
        )

    @property
    def selection(self) -> FeatureSelection:
        return FeatureSelection(layer=self.layer, head=self.head)


_FULL_BAND_KEYS = TrackingConfig(
    query_feature=FeatureKind.KEY,
    target_feature=FeatureKind.KEY,
    update_feature=FeatureKind.KEY,
)


DAVIS_PROFILES = {
    "wan": DavisProfile(
        model=WAN_2_1,
        layer=15,
        head=2,
        tracking=TrackingConfig(
            query_feature=FeatureKind.KEY,
            target_feature=FeatureKind.KEY,
            update_feature=FeatureKind.KEY,
            argmax_radius=17.0,
            search_radius=42.0,
            visibility_threshold=18.0,
            feature_ema_alpha=0.05,
            frequency_range=(0.15, 1.0),
        ),
    ),
    "cosmos2": DavisProfile(
        model=COSMOS_2,
        layer=18,
        head=6,
        tracking=TrackingConfig(
            query_feature=FeatureKind.KEY,
            target_feature=FeatureKind.KEY,
            update_feature=FeatureKind.KEY,
            argmax_radius=18.0,
            search_radius=42.0,
            visibility_threshold=26.0,
            feature_ema_alpha=0.05,
            frequency_range=(0.1, 1.0),
        ),
    ),
    "cogvideox": DavisProfile(
        model=COGVIDEOX,
        layer=15,
        head=16,
        tracking=TrackingConfig(
            query_feature=FeatureKind.KEY,
            target_feature=FeatureKind.KEY,
            update_feature=FeatureKind.KEY,
            argmax_radius=13.0,
            search_radius=43.0,
            visibility_threshold=23.0,
            feature_ema_alpha=0.05,
        ),
    ),
    "wan2.2-ti2v-5b": DavisProfile(
        model=WAN_2_2_TI2V_5B, layer=18, head=0, tracking=_FULL_BAND_KEYS
    ),
    "cogvideox1.5-5b": DavisProfile(
        model=COGVIDEOX_1_5_5B, layer=18, head=0, tracking=_FULL_BAND_KEYS
    ),
    "cosmos3-edge": DavisProfile(
        model=COSMOS_3_EDGE, layer=18, head=0, tracking=_FULL_BAND_KEYS
    ),
}
