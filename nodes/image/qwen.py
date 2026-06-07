"""Alibaba Qwen image nodes (via Kie.ai).

Covers all 5 Qwen image endpoints in Kie.ai's Market:

- qwen/text-to-image
- qwen/image-to-image
- qwen/image-edit
- qwen2/text-to-image
- qwen2/image-edit

Per docs.kie.ai cURL examples:

Qwen v1:
- T2I: prompt, image_size ("16:9"), seed, num_inference_steps, guidance_scale,
       negative_prompt, acceleration, output_format, enable_safety_checker
- I2I: prompt, image_url (singular), strength, +above
- Edit: prompt, image_url, +above

Qwen v2 (newer):
- T2I: prompt, image_size ("16:9"), seed, output_format
- Edit: prompt, image_url, seed, output_format

Qwen2 has a simpler API surface than Qwen v1.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode


# Qwen accepts aspect-ratio strings as ``image_size`` (not enum-like for v2).
_QWEN_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
_QWEN_FORMATS = ["png", "jpeg", "webp"]
_QWEN_ACCELERATIONS = ["none", "regular", "high"]


# ============================================================ Qwen v1

class QwenT2I(BaseKieMarketImageNode):
    """Qwen v1 text-to-image (Alibaba multimodal generation)."""

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
                "num_inference_steps": ("INT", {
                    "default": 30, "min": 10, "max": 50, "step": 1,
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
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
    """Qwen v1 image-to-image (transform with strength control).

    Per docs cURL: prompt, image_url (SINGULAR), strength.
    """

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
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Single source image URL (Qwen v1 uses singular).",
                }),
                "strength": ("FLOAT", {
                    "default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "0.0 = preserve source / 1.0 = full transformation.",
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
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Qwen I2I requires image_url.")

        return {
            "prompt": kwargs["prompt"],
            "image_url": img,
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
    """Qwen v1 image-edit (precise instruction-based editing).

    Distinct from I2I: Edit is for surgical changes (add/remove/swap elements)
    while I2I is for full style transfers.
    """

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
                "image_url": ("STRING", {"default": "", "tooltip": "Source image URL."}),
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
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Qwen Image Edit requires image_url.")

        return {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "negative_prompt": kwargs.get("negative_prompt", "") or "",
            "num_inference_steps": int(kwargs.get("num_inference_steps", 30)),
            "guidance_scale": float(kwargs.get("guidance_scale", 2.5)),
            "seed": int(kwargs.get("seed") or 0),
            "output_format": kwargs.get("output_format", "png"),
        }


# ============================================================ Qwen v2

class Qwen2T2I(BaseKieMarketImageNode):
    """Qwen v2 text-to-image (next-gen, simpler API surface).

    Per docs cURL: prompt, image_size, seed, output_format.
    """

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
    """Qwen v2 image-edit (next-gen editing).

    Note: body schema inferred from Qwen v1 Edit + Qwen v2 T2I patterns.
    """

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
                "image_url": ("STRING", {"default": "", "tooltip": "Source image URL."}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "output_format": (_QWEN_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Qwen2 Image Edit requires image_url.")

        return {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "seed": int(kwargs.get("seed") or 0),
            "output_format": kwargs.get("output_format", "png"),
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieQwenT2I": QwenT2I,
    "GenesisKieQwenI2I": QwenI2I,
    "GenesisKieQwenImageEdit": QwenImageEdit,
    "GenesisKieQwen2T2I": Qwen2T2I,
    "GenesisKieQwen2ImageEdit": Qwen2ImageEdit,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieQwenT2I": "Kie — Qwen (T2I)",
    "GenesisKieQwenI2I": "Kie — Qwen (I2I)",
    "GenesisKieQwenImageEdit": "Kie — Qwen Image Edit",
    "GenesisKieQwen2T2I": "Kie — Qwen2 (T2I)",
    "GenesisKieQwen2ImageEdit": "Kie — Qwen2 Image Edit",
}
