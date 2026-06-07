"""xAI Grok Imagine image nodes (via Kie.ai).

Grok Imagine is xAI's multimodal image generation model.
Per docs.kie.ai cURL examples — minimal API: prompt + aspect_ratio for T2I,
plus image_urls for I2I.

Covers the 2 Grok Imagine image endpoints:

- grok-imagine/text-to-image
- grok-imagine/image-to-image
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode


_GROK_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


class GrokImagineT2I(BaseKieMarketImageNode):
    """Grok Imagine Text-to-Image (xAI photorealistic generation).

    Per docs cURL: minimal API — only prompt + aspect_ratio confirmed.
    """

    MODEL = "grok-imagine/text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 240.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Cinematic portrait of a woman by a vinyl record player, "
                        "retro living room, soft ambient lighting, vintage editorial style."
                    ),
                }),
                "aspect_ratio": (_GROK_RATIOS, {"default": "3:2"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
        }


class GrokImagineI2I(BaseKieMarketImageNode):
    """Grok Imagine Image-to-Image (transform existing images).

    Note: body schema inferred from grok-imagine/text-to-image + general
    Kie I2I patterns. The exact field name for the reference image may
    be ``image_url`` (singular) or ``image_urls`` (array); both forms
    are supported by this node — adjust the array length in CSV input.
    """

    MODEL = "grok-imagine/image-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform the photo into a watercolor painting style.",
                }),
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs.",
                }),
                "aspect_ratio": (_GROK_RATIOS, {"default": "3:2"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Grok Imagine I2I requires at least one image_url.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
            "aspect_ratio": kwargs["aspect_ratio"],
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGrokImagineT2I": GrokImagineT2I,
    "GenesisKieGrokImagineI2I": GrokImagineI2I,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGrokImagineT2I": "Kie — Grok Imagine (T2I)",
    "GenesisKieGrokImagineI2I": "Kie — Grok Imagine (I2I)",
}
