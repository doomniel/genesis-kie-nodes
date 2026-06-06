"""Video-generation nodes for Genesis-Kie."""
from __future__ import annotations

from . import (
    grok,
    happyhorse,
    hailuo,
    kling,
    seedance,
    sora,
    utils,
    veo31,
    wan,
    runway,        # NEW in batch 3: Runway Gen-4 Turbo + Aleph (3 nodes)
    gemini_omni,   # NEW in batch 3: Gemini Omni Video + Audio + Character (3 nodes)
)

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (
    veo31, kling, seedance, hailuo, happyhorse, grok, wan, sora, utils,
    runway, gemini_omni,
):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
