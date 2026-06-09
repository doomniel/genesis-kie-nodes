"""Alibaba Qwen image nodes (via GenesisLab proxy / Kie.ai).

Qwen v1 + v2. All I2I/Edit variants accept a single IMAGE input
(qwen uses singular ``image_url`` body field).
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_QWEN_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
_QWEN_FORMATS = ["png", "jpeg", "webp"]
_QWEN_ACCELERATIONS = ["none", "regular", "high"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


class QwenT2I(BaseKieMarketImageNode):
    MODEL = "qwen/text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A photorealistic portrait of a craftsperson in their workshop.",
                }),
                "image_size": (_QWEN_RATIOS, {"default": "1:1"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 10, "max": 50, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, ugly"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "acceleration": (_QWEN_ACCELERATIONS, {"default": "none"}),
                "output_format": (_QWEN_FORMATS, {"default": "png"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_size": kwargs["image_size"],
            "negative_prompt": kwargs.get("negative_prompt", "") or "",
            "num_inference_steps": int(kwargs["num_inference_steps"]),
            "guidance_scale": float(kwargs["guidance_scale"]),
            "seed": int(kwargs.get("seed") or 0),
            "acceleration": kwargs.get("acceleration", "none"),
            "output_format": kwargs.get("output_format", "png"),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
        }


class QwenI2I(BaseKieMarketImageNode):
    MODEL = "qwen/image-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Apply impressionist painting style to the subject.",
                }),
                "image": ("IMAGE", {"tooltip": "Source image."}),
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, ugly"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 10, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "acceleration": (_QWEN_ACCELERATIONS, {"default": "none"}),
                "output_format": (_QWEN_FORMATS, {"default": "png"}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        url = _upload_first(kwargs.get("image"))
        return {
            "prompt": kwargs["prompt"],
            "image_url": url,
            "strength": float(kwargs["strength"]),
            "negative_prompt": kwargs.get("negative_prompt", "") or "",
            "num_inference_steps": int(kwargs.get("num_inference_steps", 30)),
            "guidance_scale": float(kwargs.get("guidance_scale", 2.5)),
            "seed": int(kwargs.get("seed") or 0),
            "acceleration": kwargs.get("acceleration", "none"),
            "output_format": kwargs.get("output_format", "png"),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
        }


class QwenImageEdit(BaseKieMarketImageNode):
    MODEL = "qwen/image-edit"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 360.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Change the sky to a sunset with orange and pink clouds.",
                }),
                "image": ("IMAGE", {"tooltip": "Source image."}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "num_inference_steps": ("INT", {"default": 30, "min": 10, "max": 50}),
                "guidance_scale": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "output_format": (_QWEN_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        url = _upload_first(kwargs.get("image"))
        return {
            "prompt": kwargs["prompt"],
            "image_url": url,
            "negative_prompt": kwargs.get("negative_prompt", "") or "",
            "num_inference_steps": int(kwargs.get("num_inference_steps", 30)),
            "guidance_scale": float(kwargs.get("guidance_scale", 2.5)),
            "seed": int(kwargs.get("seed") or 0),
            "output_format": kwargs.get("output_format", "png"),
        }


class Qwen2T2I(BaseKieMarketImageNode):
    MODEL = "qwen2/text-to-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A serene Japanese garden in cherry blossom season.",
                }),
                "image_size": (_QWEN_RATIOS, {"default": "16:9"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "output_format": (_QWEN_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_size": kwargs["image_size"],
            "seed": int(kwargs.get("seed") or 0),
            "output_format": kwargs.get("output_format", "png"),
        }


class Qwen2ImageEdit(BaseKieMarketImageNode):
    MODEL = "qwen2/image-edit"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 360.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Add a small dog walking beside the person in the image.",
                }),
                "image": ("IMAGE", {"tooltip": "Source image."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "output_format": (_QWEN_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        url = _upload_first(kwargs.get("image"))
        return {
            "prompt": kwargs["prompt"],
            "image_url": url,
            "seed": int(kwargs.get("seed") or 0),
            "output_format": kwargs.get("output_format", "png"),
        }


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieQwenT2I": QwenT2I,
    "GenesisKieQwenI2I": QwenI2I,
    "GenesisKieQwenImageEdit": QwenImageEdit,
    "GenesisKieQwen2T2I": Qwen2T2I,
    "GenesisKieQwen2ImageEdit": Qwen2ImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieQwenT2I": "Qwen (T2I)",
    "GenesisKieQwenI2I": "Qwen (I2I)",
    "GenesisKieQwenImageEdit": "Qwen Image Edit",
    "GenesisKieQwen2T2I": "Qwen2 (T2I)",
    "GenesisKieQwen2ImageEdit": "Qwen2 Image Edit",
}
