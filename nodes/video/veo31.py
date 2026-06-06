"""Veo 3.1 video generation nodes.

Three tiers are exposed, all under Kie.ai's ``google/`` namespace:

- ``google/veo-3.1-lite``     → cheapest, no audio in some configs
- ``google/veo-3.1-fast``     → balanced quality / cost
- ``google/veo-3.1-quality``  → top tier, 4K + audio

Pricing reference (Kie.ai, 2026):
    Lite 720p / 8s audio       $0.15  /video
    Fast 720p / 8s audio       $0.30  /video
    Fast 4K / 8s audio         $0.90  /video
    Quality 1080p / 8s audio   $1.275 /video
    Quality 4K / 8s audio      $1.85  /video
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieVideoNode


# Resolutions accepted by all three tiers. Not every tier supports every
# resolution; Kie will reject invalid combos with a clear error.
_RESOLUTIONS = ["720p", "1080p", "4k"]
_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]
_DURATIONS = [4, 6, 8]  # seconds; Veo defaults to 8


class _Veo31Base(BaseKieVideoNode):
    """Shared INPUT_TYPES and input mapping for all Veo 3.1 tiers."""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic shot of waves at golden hour.",
                }),
                "resolution": (_RESOLUTIONS, {"default": "720p"}),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 8, "min": 4, "max": 8, "step": 2}),
                "generate_audio": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "image_url": ("STRING", {"default": "",
                              "tooltip": "Optional first-frame image URL "
                                         "for image-to-video."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
            "generate_audio": bool(kwargs.get("generate_audio", True)),
        }
        image_url = (kwargs.get("image_url") or "").strip()
        if image_url:
            body["image_url"] = image_url

        negative = (kwargs.get("negative_prompt") or "").strip()
        if negative:
            body["negative_prompt"] = negative

        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed

        return body


class Veo31Lite(_Veo31Base):
    """Cheapest Veo 3.1 tier. Great for iteration."""

    MODEL_ID = "google/veo-3.1-lite"


class Veo31Fast(_Veo31Base):
    """Balanced Veo 3.1 tier — most common production workhorse."""

    MODEL_ID = "google/veo-3.1-fast"


class Veo31Quality(_Veo31Base):
    """Top-tier Veo 3.1. Slower and more expensive; use for finals."""

    MODEL_ID = "google/veo-3.1-quality"


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieVeo31Lite": Veo31Lite,
    "GenesisKieVeo31Fast": Veo31Fast,
    "GenesisKieVeo31Quality": Veo31Quality,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieVeo31Lite": "Kie — Veo 3.1 Lite",
    "GenesisKieVeo31Fast": "Kie — Veo 3.1 Fast",
    "GenesisKieVeo31Quality": "Kie — Veo 3.1 Quality",
}
