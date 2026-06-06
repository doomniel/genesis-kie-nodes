"""Hailuo (MiniMax) video generation nodes (complete family).

Covers all 6 Hailuo endpoints in Kie.ai:

Hailuo 02 Pro:
  - 02 Pro Text-to-Video    (hailuo/02-text-to-video-pro)
  - 02 Pro Image-to-Video   (hailuo/02-image-to-video-pro)

Hailuo 02 Standard:
  - 02 Std Text-to-Video    (hailuo/02-text-to-video-standard)
  - 02 Std Image-to-Video   (hailuo/02-image-to-video-standard)

Hailuo 2.3:
  - 2.3 Pro Image-to-Video      (hailuo/02-image-to-video-pro is same endpoint)
                                 — kept under "02" naming in Kie
  - 2.3 Std Image-to-Video      (hailuo/2-3-image-to-video-standard)

Note: Kie uses "02" for the Hailuo 02 line and "2-3" for the 2.3 line.
The Pro tier of 2.3 reuses the same endpoint as 02 Pro per Kie's catalog.

All use the Market endpoint /api/v1/jobs/createTask.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


# ============================================================ Hailuo 02 Pro

class Hailuo02ProT2V(BaseKieMarketVideoNode):
    """Hailuo 02 Pro text-to-video. 5000-char prompts, prompt_optimizer."""

    MODEL = "hailuo/02-text-to-video-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic establishing shot at golden hour."}),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class Hailuo02ProI2V(BaseKieMarketVideoNode):
    """Hailuo 02 Pro image-to-video. Optional end_image_url for transition."""

    MODEL = "hailuo/02-image-to-video-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "First-frame image URL (required)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation."}),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "end_image_url": ("STRING", {"default": "", "tooltip": "Optional last-frame image URL."}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Hailuo 02 Pro I2V requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        end = (kwargs.get("end_image_url") or "").strip()
        if end:
            body["end_image_url"] = end
        return body


# ============================================================ Hailuo 02 Standard

class Hailuo02StdT2V(BaseKieMarketVideoNode):
    """Hailuo 02 Standard text-to-video. Max prompt 1500 chars, 6 or 10s."""

    MODEL = "hailuo/02-text-to-video-standard"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A comedy-style animation."}),
                "duration": (["6", "10"], {"default": "6"}),
            },
            "optional": {
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
        }


class Hailuo02StdI2V(BaseKieMarketVideoNode):
    """Hailuo 02 Standard image-to-video.

    Resolution choices: 512P or 768P (NOT 1080P — only Pro supports 1080).
    Note: 10s videos are not supported with 1080p (per docs).
    """

    MODEL = "hailuo/02-image-to-video-standard"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "First-frame image URL."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation."}),
                "duration": (["6", "10"], {"default": "10"}),
                "resolution": (["512P", "768P"], {"default": "768P"}),
            },
            "optional": {
                "end_image_url": ("STRING", {"default": "", "tooltip": "Optional last-frame URL."}),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Hailuo 02 Std I2V requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
        }
        end = (kwargs.get("end_image_url") or "").strip()
        if end:
            body["end_image_url"] = end
        return body


# ============================================================ Hailuo 2.3 Std

class Hailuo23StdI2V(BaseKieMarketVideoNode):
    """Hailuo 2.3 Standard image-to-video.

    Resolution: 768P or 1080P. Duration: 6 or 10s.
    Note: 10s videos are not supported with 1080p.
    """

    MODEL = "hailuo/2-3-image-to-video-standard"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "First-frame image URL (required)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation, dramatic lighting."}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Hailuo 2.3 Std I2V requires image_url.")
        # Validate 10s + 1080p restriction per docs
        if str(kwargs["duration"]) == "10" and kwargs["resolution"] == "1080P":
            raise ValueError("Hailuo 2.3 Std: 10s videos are not supported at 1080P resolution.")
        return {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieHailuo02ProT2V": Hailuo02ProT2V,
    "GenesisKieHailuo02ProI2V": Hailuo02ProI2V,
    "GenesisKieHailuo02StdT2V": Hailuo02StdT2V,
    "GenesisKieHailuo02StdI2V": Hailuo02StdI2V,
    "GenesisKieHailuo23StdI2V": Hailuo23StdI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieHailuo02ProT2V": "Kie — Hailuo 02 Pro (T2V)",
    "GenesisKieHailuo02ProI2V": "Kie — Hailuo 02 Pro (I2V)",
    "GenesisKieHailuo02StdT2V": "Kie — Hailuo 02 Standard (T2V)",
    "GenesisKieHailuo02StdI2V": "Kie — Hailuo 02 Standard (I2V)",
    "GenesisKieHailuo23StdI2V": "Kie — Hailuo 2.3 Standard (I2V)",
}
