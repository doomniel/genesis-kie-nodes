"""Veo 3.1 video generation nodes (dedicated API via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieVeoVideoNode
from ...client.upload import upload_image_tensor


_RESOLUTIONS = ["720p", "1080p", "4k"]
_ASPECT_RATIOS = ["16:9", "9:16", "Auto"]


def _upload_first_optional(image_tensor: Any) -> str | None:
    if image_tensor is None:
        return None
    return upload_image_tensor(image_tensor[0:1])


class _Veo31Base(BaseKieVeoVideoNode):
    MODEL: ClassVar[str] = ""

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
                "start_image": ("IMAGE", {
                    "tooltip": "Optional first-frame image (image-to-video mode).",
                }),
                "end_image": ("IMAGE", {
                    "tooltip": "Optional last-frame image. With start_image: FIRST_AND_LAST_FRAMES mode.",
                }),
                "enable_translation": ("BOOLEAN", {"default": True}),
                "watermark": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_veo_request(self, **kwargs: Any) -> dict[str, Any]:
        start_url = _upload_first_optional(kwargs.get("start_image"))
        end_url = _upload_first_optional(kwargs.get("end_image"))

        image_urls: list[str] | None = None
        if start_url and end_url:
            image_urls = [start_url, end_url]
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
        elif start_url:
            image_urls = [start_url]
            generation_type = "FIRST_AND_LAST_FRAMES_2_VIDEO"
        else:
            generation_type = "TEXT_2_VIDEO"

        watermark = (kwargs.get("watermark") or "").strip() or None
        seed = int(kwargs.get("seed") or 0)
        seeds = seed if seed > 0 else None

        return {
            "prompt": kwargs["prompt"],
            "image_urls": image_urls,
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "generation_type": generation_type,
            "enable_translation": bool(kwargs.get("enable_translation", True)),
            "watermark": watermark,
            "seeds": seeds,
        }


class Veo31Lite(_Veo31Base):
    MODEL = "veo3_lite"


class Veo31Fast(_Veo31Base):
    MODEL = "veo3_fast"


class Veo31Quality(_Veo31Base):
    MODEL = "veo3"


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieVeo31Lite": Veo31Lite,
    "GenesisKieVeo31Fast": Veo31Fast,
    "GenesisKieVeo31Quality": Veo31Quality,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieVeo31Lite": "Veo 3.1 Lite",
    "GenesisKieVeo31Fast": "Veo 3.1 Fast",
    "GenesisKieVeo31Quality": "Veo 3.1 Quality",
}
