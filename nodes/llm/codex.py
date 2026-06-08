"""OpenAI Codex node for Genesis-Kie.

1 node exposing the full GPT Codex family (5 variants behind a dropdown):
- gpt-5-codex
- gpt-5.1-codex
- gpt-5.2-codex
- gpt-5.3-codex
- gpt-5.4-codex   ← default (flagship)

All use the unified ``/api/v1/responses`` endpoint (note: this is DIFFERENT
from the ``/codex/v1/responses`` endpoint used by GPT 5.4 / 5.5 — those are
regular GPT models on the Responses API; Codex variants live under
``/api/v1/responses`` per docs.kie.ai/market/codex/gpt-codex).

Best for: code generation, debugging, agentic coding tasks (Codex is tuned
to run inside agent loops with tools). Per OpenAI, the Codex family was
trained on real-world software engineering corpora + tool-use trajectories.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieChatResponsesNode


_CODEX_VERSIONS = [
    "gpt-5-codex",
    "gpt-5.1-codex",
    "gpt-5.2-codex",
    "gpt-5.3-codex",
    "gpt-5.4-codex",
]


class GPTCodex(BaseKieChatResponsesNode):
    """GPT Codex — OpenAI's coding-specialist family (Responses API).

    Use the ``codex_version`` dropdown to pick a specific variant:
    - **gpt-5.4-codex** (default): latest flagship Codex
    - **gpt-5.3-codex**: feb 2026 release, 25% faster than Claude Opus 4.6
    - **gpt-5.2-codex**: balanced cost + quality, 400K context
    - **gpt-5.1-codex**: cheaper tier, still very capable
    - **gpt-5-codex**: original Codex, lower cost
    """

    MODEL = "gpt-codex"
    MODEL_ID = "gpt-5.4-codex"  # Default; overridden by codex_version input.
    ENDPOINT = "/api/v1/responses"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        types = super().INPUT_TYPES()
        # Add codex_version dropdown as REQUIRED so the user picks explicitly.
        types["required"]["codex_version"] = (_CODEX_VERSIONS, {
            "default": "gpt-5.4-codex",
            "tooltip": "Pick a specific Codex variant. 5.4 = latest flagship.",
        })
        return types

    def build_body(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_body(**kwargs)
        # Override model with the selected Codex version.
        version = kwargs.get("codex_version", "gpt-5.4-codex")
        if version not in _CODEX_VERSIONS:
            raise ValueError(
                f"Invalid codex_version: {version}. "
                f"Must be one of {_CODEX_VERSIONS}"
            )
        body["model"] = version
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGPTCodex": GPTCodex,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGPTCodex": "Kie — GPT Codex (coding family)",
}
