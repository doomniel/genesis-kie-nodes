"""Alibaba HappyHorse 1.0 video generation nodes.

HappyHorse 1.0 ranks #1 on Artificial Analysis for video quality among Wan
family models. Uses the Market endpoint.

Pricing reference (Kie.ai, 2026):
    HappyHorse 1.0 720p         $0.14/s
    HappyHorse 1.0 1080p        $0.24/s
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


_RESOLUTIONS = ["720p", "1080p"]
_ASPECT_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]


class _HappyHorseBase(BaseKieMarketVideoNode):
    """Shared scaffolding for HappyHorse 1.0 tiers."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A miniature cardboard city at night, "
                               "small lights glowing.",
                }),
                "resolution": (_RESOLUTIONS, {"default": "1080p"}),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                }),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31 - 1,
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class HappyHorseT2V(_HappyHorseBase):
    """HappyHorse 1.0 — text-to-video."""
    MODEL = "happyhorse/text-to-video"


class HappyHorseI2V(_HappyHorseBase):
    """HappyHorse 1.0 — image-to-video. Requires ``image_url``."""

    MODEL = "happyhorse/image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"] = {
            "image_url": ("STRING", {
                "default": "",
                "tooltip": "Input image URL (required for i2v).",
            }),
            **schema["required"],
        }
        return schema

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        image_url = (kwargs.get("image_url") or "").strip()
        if not image_url:
            raise ValueError("HappyHorse I2V requires image_url.")
        body["image_url"] = image_url
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieHappyHorseT2V": HappyHorseT2V,
    "GenesisKieHappyHorseI2V": HappyHorseI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieHappyHorseT2V": "Kie — HappyHorse 1.0 (T2V)",
    "GenesisKieHappyHorseI2V": "Kie — HappyHorse 1.0 (I2V)",
}
