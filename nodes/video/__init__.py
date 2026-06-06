"""Video-generation nodes for Genesis-Kie.

Each model lives in its own module. This file aggregates their
``NODE_CLASS_MAPPINGS`` and ``NODE_DISPLAY_NAME_MAPPINGS`` so the top-level
package can register everything at once.
"""

from __future__ import annotations

from . import veo31

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (veo31,):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
