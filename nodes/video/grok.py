"""xAI Grok Imagine video generation nodes.

Grok Imagine is uniquely strong on stylized/surreal content thanks to xAI's
different training data. The "spicy" mode is more dynamic than competitors.

Pricing reference (Kie.ai, 2026):
    Grok Imagine Video 480p     $0.008/s  (84% off fal)
    Grok Imagine Video 720p     $0.015/s  (79% off fal)
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


_ASPECT_RATIOS = ["2:3", "3:2", "1:1", "16:9", "9:16"]
_MODES = ["fun", "normal", "spicy"]
_RESOLUTIONS = ["480p", "720p"]


class _GrokImagineVideoBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Grok Imagine video."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A miniature scene comes alive with playful "
                               "motion.",
                }),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
                "mode": (_MODES, {"default": "normal"}),
                "resolution": (_RESOLUTIONS, {"default": "720p"}),
                "duration": ("INT", {
                    "default": 6, "min": 6, "max": 30, "step": 1,
                }),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "mode": kwargs["mode"],
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        return body


class GrokImagineT2V(_GrokImagineVideoBase):
    """Grok Imagine — text-to-video."""
    MODEL = "grok-imagine/text-to-video"


class GrokImagineI2V(_GrokImagineVideoBase):
    """Grok Imagine — image-to-video. Requires ``image_url``."""

    MODEL = "grok-imagine/image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"] = {
            "image_url": ("STRING", {
                "default": "",
                "tooltip": "Input image URL to animate (required).",
            }),
            **schema["required"],
        }
        return schema

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        image_url = (kwargs.get("image_url") or "").strip()
        if not image_url:
            raise ValueError("Grok Imagine I2V requires image_url.")
        body["image_url"] = image_url
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGrokImagineT2V": GrokImagineT2V,
    "GenesisKieGrokImagineI2V": GrokImagineI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGrokImagineT2V": "Kie — Grok Imagine (T2V)",
    "GenesisKieGrokImagineI2V": "Kie — Grok Imagine (I2V)",
}
