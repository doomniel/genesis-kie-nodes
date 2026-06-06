"""Hailuo (MiniMax) video generation nodes.

This batch covers Hailuo 2.3 Pro, the flagship tier:
- Text-to-video
- Image-to-video

Both use the Market endpoint ``/api/v1/jobs/createTask``.

Pricing reference (Kie.ai, 2026):
    Hailuo 2.3 Pro 6s 1080p     $0.40/video
    Hailuo 2.3 Pro 6s 768p      $0.225/video
    Hailuo 2.3 Pro 10s 768p     $0.45/video
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


class _Hailuo23ProBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Hailuo 2.3 Pro tiers."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic wide shot at golden hour.",
                }),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        return body


class Hailuo23ProT2V(_Hailuo23ProBase):
    """Hailuo 2.3 Pro — text-to-video."""

    MODEL = "hailuo/02-text-to-video-pro"
    # Note: Kie's docs call this hailuo/02-text-to-video-pro (Hailuo 02 brand),
    # but the same endpoint serves what they refer to as "Hailuo 2.3 Pro" in
    # pricing. We surface the t2v Pro tier here.


class Hailuo23ProI2V(_Hailuo23ProBase):
    """Hailuo 2.3 Pro — image-to-video. Requires ``image_url``."""

    MODEL = "hailuo/02-image-to-video-pro"

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
        schema["optional"]["end_image_url"] = ("STRING", {
            "default": "",
            "tooltip": "Optional last-frame image URL.",
        })
        return schema

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        image_url = (kwargs.get("image_url") or "").strip()
        if not image_url:
            raise ValueError("Hailuo Pro I2V requires image_url.")
        body["image_url"] = image_url
        end_url = (kwargs.get("end_image_url") or "").strip()
        if end_url:
            body["end_image_url"] = end_url
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieHailuo23ProT2V": Hailuo23ProT2V,
    "GenesisKieHailuo23ProI2V": Hailuo23ProI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieHailuo23ProT2V": "Kie — Hailuo Pro (T2V)",
    "GenesisKieHailuo23ProI2V": "Kie — Hailuo Pro (I2V)",
}
