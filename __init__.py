"""Genesis-Kie — ComfyUI custom nodes for Kie.ai.

ComfyUI auto-loads ``custom_nodes/<package>/__init__.py`` and looks for two
top-level dicts:

- ``NODE_CLASS_MAPPINGS``: {internal_id: NodeClass}
- ``NODE_DISPLAY_NAME_MAPPINGS``: {internal_id: human_label}

We delegate the actual node definitions to ``nodes/`` and re-export the
merged mappings here.
"""

from __future__ import annotations

import logging

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Configure a logger so debugging is easy from ComfyUI's console.
log = logging.getLogger("genesis_kie")
if not log.handlers:
    log.setLevel(logging.INFO)

log.info(
    "Genesis-Kie loaded: %d nodes registered.",
    len(NODE_CLASS_MAPPINGS),
)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
