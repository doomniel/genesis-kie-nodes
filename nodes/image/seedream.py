"""Bytedance Seedream image generation nodes (via Kie.ai).

Covers all 7 Seedream endpoints in Kie.ai's Market:

- bytedance/seedream                       (Seedream 3.0 T2I)
- bytedance/seedream-v4-text-to-image      (Seedream 4.0 T2I)
- bytedance/seedream-v4-edit               (Seedream 4.0 Edit, multi-ref)
- seedream/4.5-text-to-image               (Seedream 4.5 T2I)
- seedream/4.5-edit                        (Seedream 4.5 Edit)
- seedream/5-lite-text-to-image            (Seedream 5.0 Lite T2I)
- seedream/5-lite-image-to-image           (Seedream 5.0 Lite I2I)

Parameter schemas extracted verbatim from docs.kie.ai cURL examples.

The Seedream family has TWO distinct parameter conventions:
- 3.0/4.0: use ``image_size`` ("square_hd", etc) + ``image_resolution`` ("1K"/"2K")
- 4.5/5.0 Lite: use ``aspect_ratio`` ("1:1", etc) + ``quality`` ("basic"/"high")

The 4.5+ "basic" vs "high" controls output resolution tier:
- 4.5: basic=2K, high=4K
- 5.0 Lite: basic=2K, high=3K
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode


# Parameter enums — exact values from docs.kie.ai cURL examples.

_V3_IMAGE_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9",
                   "landscape_4_3", "landscape_16_9"]
_V4_RESOLUTIONS = ["1K", "2K"]

_45_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "21:9", "3:2", "2:3"]
_45_QUALITY = ["basic", "high"]


def _csv(value: str) -> list[str]:
    """Split a comma-separated string into a list of trimmed values."""
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Seedream 3.0

class Seedream3T2I(BaseKieMarketImageNode):
    """Seedream 3.0 text-to-image (native 2K, fast bilingual text rendering)."""

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
                "guidance_scale": ("FLOAT", {
                    "default": 2.5, "min": 0.0, "max": 10.0, "step": 0.1,
                }),
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


# ============================================================ Seedream 4.0

class Seedream4T2I(BaseKieMarketImageNode):
    """Seedream 4.0 text-to-image (improved consistency, multi-image batch)."""

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
    """Seedream 4.0 image editing (multi-reference, character consistency).

    Per docs: supports up to 4 reference images for multi-image fusion.
    """

    MODEL = "bytedance/seedream-v4-edit"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Refer to the logo and create a brand showcase with hat, bag, wristband.",
                }),
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (1-4).",
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
        imgs = _csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Seedream 4 Edit requires at least one image_url.")
        if len(imgs) > 4:
            raise ValueError(f"Seedream 4 Edit: max 4 images, got {len(imgs)}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
            "image_size": kwargs["image_size"],
            "image_resolution": kwargs["image_resolution"],
            "max_images": int(kwargs["max_images"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", True)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ============================================================ Seedream 4.5

class Seedream45T2I(BaseKieMarketImageNode):
    """Seedream 4.5 text-to-image (4K output, refined details)."""

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
                "quality": (_45_QUALITY, {
                    "default": "basic",
                    "tooltip": "basic=2K, high=4K (per docs).",
                }),
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
    """Seedream 4.5 image editing (multi-reference, brand consistency).

    Note: body schema inferred by extrapolation from 4.5 T2I + 4.0 Edit
    patterns. If the actual API field differs, adjust ``build_input``.
    """

    MODEL = "seedream/4.5-edit"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Combine the reference images while preserving the main subject.",
                }),
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (1-10).",
                }),
                "aspect_ratio": (_45_RATIOS, {"default": "1:1"}),
                "quality": (_45_QUALITY, {"default": "basic"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Seedream 4.5 Edit requires at least one image_url.")
        if len(imgs) > 10:
            raise ValueError(f"Seedream 4.5 Edit: max 10 images, got {len(imgs)}.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs["quality"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


# ============================================================ Seedream 5.0 Lite

class Seedream5LiteT2I(BaseKieMarketImageNode):
    """Seedream 5.0 Lite text-to-image (reasoning-aware, layout-controlled)."""

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
                "quality": (_45_QUALITY, {
                    "default": "basic",
                    "tooltip": "basic=2K, high=3K (per docs).",
                }),
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
    """Seedream 5.0 Lite image-to-image (improved instruction accuracy).

    Note: body schema inferred from 5.0 Lite T2I + 4.5 Edit patterns.
    """

    MODEL = "seedream/5-lite-image-to-image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Adjust the lighting and add a warmer color grade to the subject.",
                }),
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (1-10).",
                }),
                "aspect_ratio": (_45_RATIOS, {"default": "1:1"}),
                "quality": (_45_QUALITY, {"default": "basic"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Seedream 5.0 Lite I2I requires at least one image_url.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
            "aspect_ratio": kwargs["aspect_ratio"],
            "quality": kwargs["quality"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


# ----------------------------------------------------------------- Registration

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
    "GenesisKieSeedream3T2I": "Kie — Seedream 3.0 (T2I)",
    "GenesisKieSeedream4T2I": "Kie — Seedream 4.0 (T2I)",
    "GenesisKieSeedream4Edit": "Kie — Seedream 4.0 Edit",
    "GenesisKieSeedream45T2I": "Kie — Seedream 4.5 (T2I)",
    "GenesisKieSeedream45Edit": "Kie — Seedream 4.5 Edit",
    "GenesisKieSeedream5LiteT2I": "Kie — Seedream 5.0 Lite (T2I)",
    "GenesisKieSeedream5LiteI2I": "Kie — Seedream 5.0 Lite (I2I)",
}
