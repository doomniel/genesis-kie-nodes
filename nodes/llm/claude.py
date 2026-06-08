"""Anthropic Claude nodes for Genesis-Kie.

7 nodes covering the full Claude 4 family available on Kie.ai:

- Opus 4.5 / 4.6 / 4.7 / 4.8  (frontier — best for hard reasoning + long agentic tasks)
- Sonnet 4.5 / 4.6             (balanced — best for writing + instruction following)
- Haiku 4.5                    (fast tier — cheap, good for high-volume tasks)

All use the same endpoint (/claude/v1/messages) with the model field
distinguishing variants. All support multimodal input (text + image_url)
and optional extended thinking blocks.

Note: when thinking=true, temperature is forced to 1.0 (Anthropic constraint).
"""

from __future__ import annotations

from ..base import BaseKieChatAnthropicNode


_CLAUDE_ENDPOINT = "/claude/v1/messages"


# -------------------------------------------------------- Opus (frontier tier)

class ClaudeOpus4_5(BaseKieChatAnthropicNode):
    """Claude Opus 4.5 — older frontier model, still very capable."""
    MODEL = "claude-opus-4-5"
    MODEL_ID = "claude-opus-4-5"
    ENDPOINT = _CLAUDE_ENDPOINT


class ClaudeOpus4_6(BaseKieChatAnthropicNode):
    """Claude Opus 4.6 — strong coding + long-form writing."""
    MODEL = "claude-opus-4-6"
    MODEL_ID = "claude-opus-4-6"
    ENDPOINT = _CLAUDE_ENDPOINT


class ClaudeOpus4_7(BaseKieChatAnthropicNode):
    """Claude Opus 4.7 — close competitor to Gemini 3.1 Pro on reasoning."""
    MODEL = "claude-opus-4-7"
    MODEL_ID = "claude-opus-4-7"
    ENDPOINT = _CLAUDE_ENDPOINT


class ClaudeOpus4_8(BaseKieChatAnthropicNode):
    """Claude Opus 4.8 — Anthropic's current frontier (released May 28, 2026).

    Best for: long agentic tasks, complex coding, multi-step reasoning.
    Per industry benchmarks, leads Intelligence Index at 61.4 (as of May 2026).
    """
    MODEL = "claude-opus-4-8"
    MODEL_ID = "claude-opus-4-8"
    ENDPOINT = _CLAUDE_ENDPOINT


# ------------------------------------------------------- Sonnet (balanced tier)

class ClaudeSonnet4_5(BaseKieChatAnthropicNode):
    """Claude Sonnet 4.5 — balanced quality/speed/cost."""
    MODEL = "claude-sonnet-4-5"
    MODEL_ID = "claude-sonnet-4-5"
    ENDPOINT = _CLAUDE_ENDPOINT


class ClaudeSonnet4_6(BaseKieChatAnthropicNode):
    """Claude Sonnet 4.6 — best for writing style + instruction following.

    Per industry consensus, the model most teams reach for in production
    when they need quality + reasonable cost (vs Opus's premium pricing).
    """
    MODEL = "claude-sonnet-4-6"
    MODEL_ID = "claude-sonnet-4-6"
    ENDPOINT = _CLAUDE_ENDPOINT


# ----------------------------------------------------------- Haiku (fast tier)

class ClaudeHaiku4_5(BaseKieChatAnthropicNode):
    """Claude Haiku 4.5 — fast + cheap, good for high-volume tasks.

    Use for: classification, summarization, simple Q&A, batch processing
    where latency + cost matter more than absolute quality.
    """
    MODEL = "claude-haiku-4-5"
    MODEL_ID = "claude-haiku-4-5"
    ENDPOINT = _CLAUDE_ENDPOINT


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieClaudeOpus4_5": ClaudeOpus4_5,
    "GenesisKieClaudeOpus4_6": ClaudeOpus4_6,
    "GenesisKieClaudeOpus4_7": ClaudeOpus4_7,
    "GenesisKieClaudeOpus4_8": ClaudeOpus4_8,
    "GenesisKieClaudeSonnet4_5": ClaudeSonnet4_5,
    "GenesisKieClaudeSonnet4_6": ClaudeSonnet4_6,
    "GenesisKieClaudeHaiku4_5": ClaudeHaiku4_5,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieClaudeOpus4_5": "Kie — Claude Opus 4.5",
    "GenesisKieClaudeOpus4_6": "Kie — Claude Opus 4.6",
    "GenesisKieClaudeOpus4_7": "Kie — Claude Opus 4.7",
    "GenesisKieClaudeOpus4_8": "Kie — Claude Opus 4.8 (frontier)",
    "GenesisKieClaudeSonnet4_5": "Kie — Claude Sonnet 4.5",
    "GenesisKieClaudeSonnet4_6": "Kie — Claude Sonnet 4.6",
    "GenesisKieClaudeHaiku4_5": "Kie — Claude Haiku 4.5 (fast)",
}
