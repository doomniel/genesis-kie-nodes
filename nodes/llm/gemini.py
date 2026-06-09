"""Google Gemini nodes for Genesis-Kie (OpenAI-compatible variants).

6 nodes, all using OpenAI Chat Completions API shape via Kie's gateway:

- Gemini 2.5 Pro    /gemini-2.5-pro/v1/chat/completions
- Gemini 2.5 Flash  /gemini-2.5-flash/v1/chat/completions
- Gemini 3 Pro      /gemini-3-pro/v1/chat/completions
- Gemini 3 Flash    /gemini-3-flash/v1/chat/completions
- Gemini 3.1 Pro    /gemini-3.1-pro/v1/chat/completions
- Gemini 3.5 Flash  /gemini-3.5-flash/v1/chat/completions

All support multimodal input (text + image_url) and the standard OpenAI
chat completions body shape. Pro variants prioritize quality, Flash
variants optimize for speed + cost.

Per industry consensus (May 2026):
- Gemini 3.1 Pro = best for hard reasoning benchmarks (94.3% GPQA Diamond)
- Gemini 3.5 Flash = best price-to-performance at the frontier

Note: Kie.ai also offers Gemini 3/3.5 Flash "native" (non-OpenAI) variants
at different endpoints. Those use Google's native format with different
multimodal handling — not exposed in this batch (v2 feature if needed).

GOTCHA: Kie's Gemini gateway REJECTS bodies that include a `model` field
("[422] The model is not supported"). The endpoint path already identifies
the model, so we strip it. This is handled by ``_GeminiChatBase`` below
which overrides ``build_body()`` to remove the field.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieChatOpenAINode


class _GeminiChatBase(BaseKieChatOpenAINode):
    """Shared base for all Gemini OpenAI-compat nodes.

    Identical to ``BaseKieChatOpenAINode`` except: ``model`` is removed
    from the request body. Kie's Gemini gateway rejects requests that
    include it because the endpoint URL (e.g. /gemini-3.5-flash/...)
    already identifies the model.
    """

    def build_body(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_body(**kwargs)
        body.pop("model", None)
        return body


# -------------------------------------------------------------------- Pro tier

class Gemini2_5Pro(_GeminiChatBase):
    """Gemini 2.5 Pro — Google's mid-tier quality + reasoning."""
    MODEL = "gemini-2.5-pro"
    MODEL_ID = "gemini-2.5-pro"
    ENDPOINT = "/gemini-2.5-pro/v1/chat/completions"


class Gemini3Pro(_GeminiChatBase):
    """Gemini 3 Pro — next-gen with improved multimodal."""
    MODEL = "gemini-3-pro"
    MODEL_ID = "gemini-3-pro"
    ENDPOINT = "/gemini-3-pro/v1/chat/completions"


class Gemini3_1Pro(_GeminiChatBase):
    """Gemini 3.1 Pro — Google's current frontier (leads GPQA Diamond at 94.3%).

    Best for: hardest reasoning tasks, research workflows, anything needing
    extreme accuracy. 1M context window. ARC-AGI-2 leader at 77.1%.
    """
    MODEL = "gemini-3.1-pro"
    MODEL_ID = "gemini-3.1-pro"
    ENDPOINT = "/gemini-3.1-pro/v1/chat/completions"


# ------------------------------------------------------------------ Flash tier

class Gemini2_5Flash(_GeminiChatBase):
    """Gemini 2.5 Flash — older fast tier, lowest cost."""
    MODEL = "gemini-2.5-flash"
    MODEL_ID = "gemini-2.5-flash"
    ENDPOINT = "/gemini-2.5-flash/v1/chat/completions"


class Gemini3Flash(_GeminiChatBase):
    """Gemini 3 Flash — fast tier with v3 improvements."""
    MODEL = "gemini-3-flash"
    MODEL_ID = "gemini-3-flash"
    ENDPOINT = "/gemini-3-flash/v1/chat/completions"


class Gemini3_5Flash(_GeminiChatBase):
    """Gemini 3.5 Flash — current fast tier (best price-to-perf at frontier).

    Per industry consensus, the sweet spot for production workloads:
    Intelligence Index 55, far cheaper than Pro tiers, very low hallucination
    rate (3.3% on Vectara — better than reasoning models which exceed 10%).
    """
    MODEL = "gemini-3.5-flash"
    MODEL_ID = "gemini-3.5-flash"
    ENDPOINT = "/gemini-3.5-flash/v1/chat/completions"


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGemini2_5Pro": Gemini2_5Pro,
    "GenesisKieGemini3Pro": Gemini3Pro,
    "GenesisKieGemini3_1Pro": Gemini3_1Pro,
    "GenesisKieGemini2_5Flash": Gemini2_5Flash,
    "GenesisKieGemini3Flash": Gemini3Flash,
    "GenesisKieGemini3_5Flash": Gemini3_5Flash,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGemini2_5Pro": "Gemini 2.5 Pro",
    "GenesisKieGemini3Pro": "Gemini 3 Pro",
    "GenesisKieGemini3_1Pro": "Gemini 3.1 Pro (frontier)",
    "GenesisKieGemini2_5Flash": "Gemini 2.5 Flash",
    "GenesisKieGemini3Flash": "Gemini 3 Flash",
    "GenesisKieGemini3_5Flash": "Gemini 3.5 Flash (best value)",
}
