"""OpenAI GPT Image (1.5 + 2) image nodes (via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_GPT_RATIOS = ["auto", "1:1", "3:2", "2:3", "16:9", "9:16"]
_GPT_QUALITY = ["auto", "low", "medium", "high"]


def _upload_batch(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class GPTImage15T2I(BaseKieMarketImageNode):
    MODEL = "gpt-image-1.5/text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A poster with the words 'GPT IMAGE 1.5' in clean modern typography.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
            "optional": {
                "quality": (_GPT_QUALITY, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs.get("quality", "auto"),
        }


class GPTImage15I2I(BaseKieMarketImageNode):
    MODEL = "gpt-image-1.5/image-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Refine the input image with cinematic color grading.",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Reference image(s). Batch for multi-ref.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
            "optional": {
                "quality": (_GPT_QUALITY, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("GPT Image 1.5 I2I requires at least one image.")

        return {
            "prompt": kwargs["prompt"],
            "input_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs.get("quality", "auto"),
        }


class GPTImage2T2I(BaseKieMarketImageNode):
    MODEL = "gpt-image-2-text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic night city poster with neon reflections on a rainy street.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
        }


class GPTImage2I2I(BaseKieMarketImageNode):
    MODEL = "gpt-image-2-image-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Add a person reading a book to the conference room scene.",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Reference image(s). Batch for multi-ref.",
                }),
                "aspect_ratio": (_GPT_RATIOS, {"default": "auto"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("GPT Image 2 I2I requires at least one image.")

        return {
            "prompt": kwargs["prompt"],
            "input_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
        }


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGPTImage15T2I": GPTImage15T2I,
    "GenesisKieGPTImage15I2I": GPTImage15I2I,
    "GenesisKieGPTImage2T2I": GPTImage2T2I,
    "GenesisKieGPTImage2I2I": GPTImage2I2I,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGPTImage15T2I": "GPT Image 1.5 (T2I)",
    "GenesisKieGPTImage15I2I": "GPT Image 1.5 (I2I)",
    "GenesisKieGPTImage2T2I": "GPT Image 2 (T2I)",
    "GenesisKieGPTImage2I2I": "GPT Image 2 (I2I)",
}
