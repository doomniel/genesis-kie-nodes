"""OpenAI GPT nodes for Genesis-Kie.

3 nodes covering both GPT API shapes:

- GPT 5.2:  Patrón A — /gpt-5-2/v1/chat/completions (OpenAI Chat Completions)
- GPT 5.4:  Patrón B — /codex/v1/responses, model=gpt-5-4   (Responses API + reasoning)
- GPT 5.5:  Patrón B — /codex/v1/responses, model=gpt-5-5   (Responses API + reasoning)

GPT 5.4 and 5.5 expose the reasoning_effort dial (minimal/low/medium/high/xhigh)
which controls how deeply the model thinks. GPT 5.2 is the cheaper baseline
without reasoning control.
"""

from __future__ import annotations

from ..base import BaseKieChatOpenAINode, BaseKieChatResponsesNode


class GPT5_2(BaseKieChatOpenAINode):
    """GPT 5.2 — OpenAI's daily-driver chat model (Chat Completions API).

    Use for: general chat, content generation, drafting. Cheap + fast.
    Supports multimodal input (text + image_url).
    """

    MODEL = "gpt-5-2"
    MODEL_ID = "gpt-5-2"
    ENDPOINT = "/gpt-5-2/v1/chat/completions"


class GPT5_4(BaseKieChatResponsesNode):
    """GPT 5.4 — Responses API with adjustable reasoning effort.

    Use for: complex reasoning, multi-step problem solving, analysis.
    Slower + more expensive than 5.2 but more accurate on hard tasks.
    """

    MODEL = "gpt-5-4"
    MODEL_ID = "gpt-5-4"
    ENDPOINT = "/codex/v1/responses"


class GPT5_5(BaseKieChatResponsesNode):
    """GPT 5.5 — OpenAI's current flagship (Responses API + reasoning).

    Use for: hardest reasoning tasks, frontier benchmarks, when accuracy
    matters more than cost. Per docs, supports web_search and function
    calling tools (not exposed yet in this node — v2 feature).
    """

    MODEL = "gpt-5-5"
    MODEL_ID = "gpt-5-5"
    ENDPOINT = "/codex/v1/responses"


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGPT5_2": GPT5_2,
    "GenesisKieGPT5_4": GPT5_4,
    "GenesisKieGPT5_5": GPT5_5,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGPT5_2": "Kie — GPT 5.2 (chat)",
    "GenesisKieGPT5_4": "Kie — GPT 5.4 (reasoning)",
    "GenesisKieGPT5_5": "Kie — GPT 5.5 (frontier)",
}
