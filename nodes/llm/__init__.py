"""LLM / Chat nodes for Genesis-Kie.

17 LLM nodes across 4 families × 3 endpoint patterns:

  GPT (3):
    GPT 5.2          OpenAI Chat Completions
    GPT 5.4          OpenAI Responses API (+ reasoning effort)
    GPT 5.5          OpenAI Responses API (+ reasoning effort)

  Claude (7):  All use /claude/v1/messages (Anthropic Messages shape)
    Opus 4.5, 4.6, 4.7, 4.8
    Sonnet 4.5, 4.6
    Haiku 4.5

  Codex (1):
    GPT Codex        OpenAI Responses API (+ reasoning effort)

  Gemini (6):  All OpenAI-compatible variants
    2.5 Pro, 2.5 Flash, 3 Pro, 3 Flash, 3.1 Pro, 3.5 Flash

Outputs: (text, tokens_used) for every node.
Inputs: system_prompt, user_prompt, image_url (multimodal), max_tokens,
        temperature, plus reasoning_effort (Responses API) or
        thinking/thinking_budget (Anthropic).

All chat endpoints are SYNCHRONOUS — single POST, 1-30s response.
"""

from __future__ import annotations

from . import claude, codex, gemini, gpt

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (gpt, claude, codex, gemini):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
