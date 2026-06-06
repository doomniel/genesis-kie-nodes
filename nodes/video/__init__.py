"""Video-generation nodes for Genesis-Kie."""
from __future__ import annotations
from . import grok, happyhorse, hailuo, kling, seedance, sora, utils, veo31, wan

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (veo31, kling, seedance, hailuo, happyhorse, grok, wan, sora, utils):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
