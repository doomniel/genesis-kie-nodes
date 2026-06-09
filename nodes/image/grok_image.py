"""xAI Grok Imagine image nodes (via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_GROK_RATIOS = ["1:1", "16:9", "9:16", "3:2", "2:3", "4:3", "3:4"]


def _upload_batch(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class GrokImagineT2I(BaseKieMarketImageNode):
    """Grok Imagine Text-to-Image (xAI photorealistic generation)."""

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
                        "retro living room, soft ambient lighting."
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
    """Grok Imagine Image-to-Image (transform existing images)."""

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
                "images": ("IMAGE", {
                    "tooltip": "Reference image(s). Batch for multi-ref.",
                }),
                "aspect_ratio": (_GROK_RATIOS, {"default": "3:2"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("Grok Imagine I2I requires at least one image.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
        }


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGrokImagineT2I": GrokImagineT2I,
    "GenesisKieGrokImagineI2I": GrokImagineI2I,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGrokImagineT2I": "Grok Imagine (T2I)",
    "GenesisKieGrokImagineI2I": "Grok Imagine (I2I)",
}
