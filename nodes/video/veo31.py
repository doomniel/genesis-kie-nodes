"""Veo 3.1 video generation nodes.

Veo 3.1 in Kie.ai uses a DEDICATED API (not the generic Market endpoint).
The model identifiers are:

- ``veo3``       → Quality (top tier)
- ``veo3_fast``  → Fast (balanced)
- ``veo3_lite``  → Lite (cheapest)

Endpoints used:
- POST /api/v1/veo/generate
- GET  /api/v1/veo/record-info?taskId=X

Pricing reference (Kie.ai, 2026):
    Lite 720p / 8s audio       $0.15  /video
    Lite 1080p / 8s audio      $0.175 /video
    Fast 720p / 8s audio       $0.30  /video
    Fast 4K / 8s audio         $0.90  /video
    Quality 1080p / 8s audio   $1.275 /video
    Quality 4K / 8s audio      $1.85  /video
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieVeoVideoNode


_RESOLUTIONS = ["720p", "1080p", "4k"]
_ASPECT_RATIOS = ["16:9", "9:16", "Auto"]


class _Veo31Base(BaseKieVeoVideoNode):
    """Shared INPUT_TYPES + build_veo_request for all Veo 3.1 tiers."""

    MODEL: ClassVar[str] = ""  # subclasses set this

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
            },
            "optional": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional first-frame image URL for image-to-video.",
                }),
                "last_frame_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional last-frame URL. If set with image_url, "
                               "uses FIRST_AND_LAST_FRAMES_2_VIDEO mode.",
                }),
                "enable_translation": ("BOOLEAN", {"default": True}),
                "watermark": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_veo_request(self, **kwargs: Any) -> dict[str, Any]:
        prompt: str = kwargs["prompt"]
        resolution: str = kwargs["resolution"]
        aspect_ratio: str = kwargs["aspect_ratio"]

        image_url = (kwargs.get("image_url") or "").strip()
        last_frame_url = (kwargs.get("last_frame_url") or "").strip()

        image_urls: list[str] | None = None
        generation_type: str
        if image_url and last_frame_url:
            image_urls = [image_url, last_frame_url]
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
        elif image_url:
            image_urls = [image_url]
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
        else:
            generation_type = "TEXT_2_VIDEO"

        watermark = (kwargs.get("watermark") or "").strip() or None
        seed = int(kwargs.get("seed") or 0)
        seeds = seed if seed > 0 else None

        return {
            "prompt": prompt,
            "image_urls": image_urls,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "generation_type": generation_type,
            "enable_translation": bool(kwargs.get("enable_translation", True)),
            "watermark": watermark,
            "seeds": seeds,
        }


class Veo31Lite(_Veo31Base):
    """Veo 3.1 Lite — cheapest tier (~$0.15/8s 720p)."""
    MODEL = "veo3_lite"


class Veo31Fast(_Veo31Base):
    """Veo 3.1 Fast — balanced quality/cost (~$0.30/8s 720p)."""
    MODEL = "veo3_fast"


class Veo31Quality(_Veo31Base):
    """Veo 3.1 Quality — flagship tier (~$1.25/8s 720p)."""
    MODEL = "veo3"


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
