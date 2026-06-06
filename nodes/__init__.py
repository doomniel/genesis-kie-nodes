"""Aggregates node mappings from all modality submodules.

Each modality module exports ``NODE_CLASS_MAPPINGS`` and
``NODE_DISPLAY_NAME_MAPPINGS`` dicts. We merge them here so the top-level
package only needs to import this one file.
"""

from __future__ import annotations

from . import image, llm, music, video

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (video, image, music, llm):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
