"""xAI Grok Imagine video generation nodes (via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor


_ASPECT_RATIOS = ["2:3", "3:2", "1:1", "16:9", "9:16"]
_MODES = ["fun", "normal", "spicy"]
_RESOLUTIONS = ["480p", "720p"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


class _GrokImagineVideoBase(BaseKieMarketVideoNode):
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A miniature scene comes alive with playful motion.",
                }),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
                "mode": (_MODES, {"default": "normal"}),
                "resolution": (_RESOLUTIONS, {"default": "720p"}),
                "duration": ("INT", {"default": 6, "min": 6, "max": 30, "step": 1}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "mode": kwargs["mode"],
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class GrokImagineT2V(_GrokImagineVideoBase):
    MODEL = "grok-imagine/text-to-video"


class GrokImagineI2V(_GrokImagineVideoBase):
    """Grok Imagine I2V. Requires an image to animate."""
    MODEL = "grok-imagine/image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"] = {
            "image": ("IMAGE", {"tooltip": "Input image to animate."}),
            **schema["required"],
        }
        return schema

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        body["image_url"] = _upload_first(kwargs.get("image"))
        return body


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGrokImagineT2V": GrokImagineT2V,
    "GenesisKieGrokImagineI2V": GrokImagineI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGrokImagineT2V": "Grok Imagine (T2V)",
    "GenesisKieGrokImagineI2V": "Grok Imagine (I2V)",
}
