"""Bytedance Seedream image generation nodes (via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_V3_IMAGE_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9",
                   "landscape_4_3", "landscape_16_9"]
_V4_RESOLUTIONS = ["1K", "2K"]
_45_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "3:2", "2:3"]
_45_QUALITY = ["basic", "high"]


def _upload_batch(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class Seedream3T2I(BaseKieMarketImageNode):
    MODEL = "bytedance/seedream"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A 2D flat art style poster with the text 'Kie AI Seedream 3.0'.",
                }),
                "image_size": (_V3_IMAGE_SIZES, {"default": "square_hd"}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_size": kwargs["image_size"],
            "guidance_scale": float(kwargs["guidance_scale"]),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
        }


class Seedream4T2I(BaseKieMarketImageNode):
    MODEL = "bytedance/seedream-v4-text-to-image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A detailed system of binary linear equations drawn on a blackboard.",
                }),
                "image_size": (_V3_IMAGE_SIZES, {"default": "square_hd"}),
                "image_resolution": (_V4_RESOLUTIONS, {"default": "1K"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "nsfw_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_size": kwargs["image_size"],
            "image_resolution": kwargs["image_resolution"],
            "max_images": int(kwargs["max_images"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", True)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Seedream4Edit(BaseKieMarketImageNode):
    MODEL = "bytedance/seedream-v4-edit"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Refer to the logo and create a brand showcase with hat, bag, wristband.",
                }),
                "images": ("IMAGE", {
                    "tooltip": "Reference image(s). Batch for multi-ref (max 4).",
                }),
                "image_size": (_V3_IMAGE_SIZES, {"default": "square_hd"}),
                "image_resolution": (_V4_RESOLUTIONS, {"default": "1K"}),
                "max_images": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "nsfw_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("Seedream 4 Edit requires at least one image.")
        if len(urls) > 4:
            raise ValueError(f"Seedream 4 Edit: max 4 images, got {len(urls)}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_urls": urls,
            "image_size": kwargs["image_size"],
            "image_resolution": kwargs["image_resolution"],
            "max_images": int(kwargs["max_images"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", True)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Seedream45T2I(BaseKieMarketImageNode):
    MODEL = "seedream/4.5-text-to-image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cafe design tool promotional image with multi-panel composition.",
                }),
                "aspect_ratio": (_45_RATIOS, {"default": "1:1"}),
                "quality": (_45_QUALITY, {"default": "basic"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs["quality"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class Seedream45Edit(BaseKieMarketImageNode):
    MODEL = "seedream/4.5-edit"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Combine the reference images while preserving the main subject.",
                }),
                "images": ("IMAGE", {"tooltip": "Reference image(s). Batch for multi-ref (max 10)."}),
                "aspect_ratio": (_45_RATIOS, {"default": "1:1"}),
                "quality": (_45_QUALITY, {"default": "basic"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("Seedream 4.5 Edit requires at least one image.")
        if len(urls) > 10:
            raise ValueError(f"Seedream 4.5 Edit: max 10 images, got {len(urls)}.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs["quality"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class Seedream5LiteT2I(BaseKieMarketImageNode):
    MODEL = "seedream/5-lite-text-to-image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "An infographic explaining a product's value proposition.",
                }),
                "aspect_ratio": (_45_RATIOS, {"default": "1:1"}),
                "quality": (_45_QUALITY, {"default": "basic"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs["quality"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class Seedream5LiteI2I(BaseKieMarketImageNode):
    MODEL = "seedream/5-lite-image-to-image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Adjust the lighting and add a warmer color grade to the subject.",
                }),
                "images": ("IMAGE", {"tooltip": "Reference image(s). Batch for multi-ref."}),
                "aspect_ratio": (_45_RATIOS, {"default": "1:1"}),
                "quality": (_45_QUALITY, {"default": "basic"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("Seedream 5.0 Lite I2I requires at least one image.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs["quality"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSeedream3T2I": Seedream3T2I,
    "GenesisKieSeedream4T2I": Seedream4T2I,
    "GenesisKieSeedream4Edit": Seedream4Edit,
    "GenesisKieSeedream45T2I": Seedream45T2I,
    "GenesisKieSeedream45Edit": Seedream45Edit,
    "GenesisKieSeedream5LiteT2I": Seedream5LiteT2I,
    "GenesisKieSeedream5LiteI2I": Seedream5LiteI2I,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSeedream3T2I": "Seedream 3.0 (T2I)",
    "GenesisKieSeedream4T2I": "Seedream 4.0 (T2I)",
    "GenesisKieSeedream4Edit": "Seedream 4.0 Edit",
    "GenesisKieSeedream45T2I": "Seedream 4.5 (T2I)",
    "GenesisKieSeedream45Edit": "Seedream 4.5 Edit",
    "GenesisKieSeedream5LiteT2I": "Seedream 5.0 Lite (T2I)",
    "GenesisKieSeedream5LiteI2I": "Seedream 5.0 Lite (I2I)",
}
