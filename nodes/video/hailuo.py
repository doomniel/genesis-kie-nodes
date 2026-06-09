"""Hailuo (MiniMax) video generation nodes (via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


def _upload_first_optional(image_tensor: Any) -> str | None:
    if image_tensor is None:
        return None
    return upload_image_tensor(image_tensor[0:1])


class Hailuo02ProT2V(BaseKieMarketVideoNode):
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
    MODEL = "hailuo/02-image-to-video-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "First-frame image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation."}),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "end_image": ("IMAGE", {"tooltip": "Optional last-frame image."}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        end_url = _upload_first_optional(kwargs.get("end_image"))
        if end_url:
            body["end_image_url"] = end_url
        return body


class Hailuo02StdT2V(BaseKieMarketVideoNode):
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
    MODEL = "hailuo/02-image-to-video-standard"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "First-frame image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation."}),
                "duration": (["6", "10"], {"default": "10"}),
                "resolution": (["512P", "768P"], {"default": "768P"}),
            },
            "optional": {
                "end_image": ("IMAGE", {"tooltip": "Optional last-frame image."}),
                "prompt_optimizer": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "prompt_optimizer": bool(kwargs.get("prompt_optimizer", True)),
        }
        end_url = _upload_first_optional(kwargs.get("end_image"))
        if end_url:
            body["end_image_url"] = end_url
        return body


class Hailuo23StdI2V(BaseKieMarketVideoNode):
    MODEL = "hailuo/2-3-image-to-video-standard"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "First-frame image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation."}),
                "duration": (["6", "10"], {"default": "6"}),
                "resolution": (["768P", "1080P"], {"default": "768P"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        if str(kwargs["duration"]) == "10" and kwargs["resolution"] == "1080P":
            raise ValueError("Hailuo 2.3 Std: 10s videos are not supported at 1080P.")
        return {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieHailuo02ProT2V": Hailuo02ProT2V,
    "GenesisKieHailuo02ProI2V": Hailuo02ProI2V,
    "GenesisKieHailuo02StdT2V": Hailuo02StdT2V,
    "GenesisKieHailuo02StdI2V": Hailuo02StdI2V,
    "GenesisKieHailuo23StdI2V": Hailuo23StdI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieHailuo02ProT2V": "Hailuo 02 Pro (T2V)",
    "GenesisKieHailuo02ProI2V": "Hailuo 02 Pro (I2V)",
    "GenesisKieHailuo02StdT2V": "Hailuo 02 Standard (T2V)",
    "GenesisKieHailuo02StdI2V": "Hailuo 02 Standard (I2V)",
    "GenesisKieHailuo23StdI2V": "Hailuo 2.3 Standard (I2V)",
}
