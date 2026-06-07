"""OpenAI GPT Image (1.5 + 2) image nodes (via Kie.ai).

GPT Image is OpenAI's image generation family on Kie.ai's Market.
GPT Image 1.5 is OpenAI's flagship with strong instruction following and
improved text rendering. GPT Image 2 is the next-gen tier with stronger
photorealism and cleaner editing.

Covers all 4 GPT Image endpoints:

- gpt-image-1.5/text-to-image
- gpt-image-1.5/image-to-image
- gpt-image-2-text-to-image            (note: hyphen-only, no slash)
- gpt-image-2-image-to-image

Note the inconsistent model slug — 1.5 uses ``gpt-image-1.5/...``,
while 2 uses ``gpt-image-2-...``. Verified from docs.kie.ai cURL examples.

Per docs: minimum body is prompt + aspect_ratio (often "auto" supported).
GPT Image 2 supports nVariants (1/2/4) for batch generation.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode


_GPT_RATIOS = ["auto", "1:1", "3:2", "2:3", "16:9", "9:16"]
_GPT_QUALITY = ["auto", "low", "medium", "high"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ GPT Image 1.5

class GPTImage15T2I(BaseKieMarketImageNode):
    """OpenAI GPT Image 1.5 text-to-image (strong text rendering)."""

    MODEL = "gpt-image-1.5/text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A poster with the words 'GPT IMAGE 1.5' in clean modern typography.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
            "optional": {
                "quality": (_GPT_QUALITY, {
                    "default": "auto",
                    "tooltip": "auto/low/medium/high (higher = better fidelity, more cost).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs.get("quality", "auto"),
        }


class GPTImage15I2I(BaseKieMarketImageNode):
    """OpenAI GPT Image 1.5 image-to-image (edit with reference)."""

    MODEL = "gpt-image-1.5/image-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Refine the input image with cinematic color grading.",
                }),
                "input_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
            "optional": {
                "quality": (_GPT_QUALITY, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("input_urls") or "").strip())
        if not imgs:
            raise ValueError("GPT Image 1.5 I2I requires at least one input_url.")

        return {
            "prompt": kwargs["prompt"],
            "input_urls": imgs,
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs.get("quality", "auto"),
        }


# ============================================================ GPT Image 2

class GPTImage2T2I(BaseKieMarketImageNode):
    """OpenAI GPT Image 2 text-to-image (next-gen photorealism)."""

    MODEL = "gpt-image-2-text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic night city poster with neon reflections on a rainy street.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
        }


class GPTImage2I2I(BaseKieMarketImageNode):
    """OpenAI GPT Image 2 image-to-image (edit with reference)."""

    MODEL = "gpt-image-2-image-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Add a person reading a book to the conference room scene.",
                }),
                "input_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("input_urls") or "").strip())
        if not imgs:
            raise ValueError("GPT Image 2 I2I requires at least one input_url.")

        return {
            "prompt": kwargs["prompt"],
            "input_urls": imgs,
            "aspect_ratio": kwargs["aspect_ratio"],
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGPTImage15T2I": GPTImage15T2I,
    "GenesisKieGPTImage15I2I": GPTImage15I2I,
    "GenesisKieGPTImage2T2I": GPTImage2T2I,
    "GenesisKieGPTImage2I2I": GPTImage2I2I,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGPTImage15T2I": "Kie — GPT Image 1.5 (T2I)",
    "GenesisKieGPTImage15I2I": "Kie — GPT Image 1.5 (I2I)",
    "GenesisKieGPTImage2T2I": "Kie — GPT Image 2 (T2I)",
    "GenesisKieGPTImage2I2I": "Kie — GPT Image 2 (I2I)",
}
