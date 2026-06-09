"""Ideogram image nodes (via GenesisLab proxy / Kie.ai).

Ideogram has the most complex input shapes of the catalog:
- V3 Edit: image (target) + mask
- V3 Remix: image + image_weight
- Character: reference_images (batch)
- CharacterEdit: image + mask + reference_images
- CharacterRemix: image + reference_images + image_weight
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_IDEOGRAM_SPEEDS = ["TURBO", "BALANCED", "QUALITY"]
_IDEOGRAM_STYLES = ["AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"]
_IDEOGRAM_NUM_IMAGES = ["1", "2", "3", "4", "5", "6", "7", "8"]
_IDEOGRAM_V3_SIZES = [
    "1:1",
    "landscape_4_3", "landscape_3_2", "landscape_16_9",
    "portrait_4_3", "portrait_3_2", "portrait_16_9",
]
_IDEOGRAM_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


def _upload_batch(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class IdeogramV3T2I(BaseKieMarketImageNode):
    MODEL = "ideogram-v3/text-to-image"
    POLL_INTERVAL_SECONDS = 2.5
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A poster for 'Summer Design Conference' with bold modern "
                        "typography, vibrant gradient background, professional layout."
                    ),
                }),
                "image_size": (_IDEOGRAM_V3_SIZES, {"default": "1:1"}),
                "rendering_speed": (_IDEOGRAM_SPEEDS, {"default": "BALANCED"}),
                "style": (_IDEOGRAM_STYLES, {"default": "AUTO"}),
                "num_images": (_IDEOGRAM_NUM_IMAGES, {"default": "1"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "expand_prompt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_size": kwargs["image_size"],
            "rendering_speed": kwargs["rendering_speed"],
            "style": kwargs["style"],
            "num_images": str(kwargs["num_images"]),
            "expand_prompt": bool(kwargs.get("expand_prompt", True)),
        }
        neg = kwargs.get("negative_prompt", "") or ""
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class IdeogramV3Edit(BaseKieMarketImageNode):
    MODEL = "ideogram-v3/edit"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 360.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Replace the background text with 'SALE 50% OFF' in elegant typography.",
                }),
                "image": ("IMAGE", {"tooltip": "Source image."}),
                "mask": ("IMAGE", {"tooltip": "Mask (white = edit, black = preserve)."}),
                "rendering_speed": (_IDEOGRAM_SPEEDS, {"default": "BALANCED"}),
            },
            "optional": {
                "style": (_IDEOGRAM_STYLES, {"default": "AUTO"}),
                "num_images": (_IDEOGRAM_NUM_IMAGES, {"default": "1"}),
                "expand_prompt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img_url = _upload_first(kwargs.get("image"))
        mask_url = _upload_first(kwargs.get("mask"))

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img_url,
            "mask_url": mask_url,
            "rendering_speed": kwargs["rendering_speed"],
            "style": kwargs.get("style", "AUTO"),
            "num_images": str(kwargs.get("num_images", "1")),
            "expand_prompt": bool(kwargs.get("expand_prompt", True)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class IdeogramV3Remix(BaseKieMarketImageNode):
    MODEL = "ideogram-v3/remix"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Recompose with a watercolor style and warmer color palette.",
                }),
                "image": ("IMAGE", {"tooltip": "Source image to remix."}),
                "image_weight": ("INT", {"default": 50, "min": 0, "max": 100, "step": 5}),
                "rendering_speed": (_IDEOGRAM_SPEEDS, {"default": "BALANCED"}),
            },
            "optional": {
                "style": (_IDEOGRAM_STYLES, {"default": "AUTO"}),
                "num_images": (_IDEOGRAM_NUM_IMAGES, {"default": "1"}),
                "expand_prompt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img_url = _upload_first(kwargs.get("image"))

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img_url,
            "image_weight": int(kwargs["image_weight"]),
            "rendering_speed": kwargs["rendering_speed"],
            "style": kwargs.get("style", "AUTO"),
            "num_images": str(kwargs.get("num_images", "1")),
            "expand_prompt": bool(kwargs.get("expand_prompt", True)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class IdeogramCharacter(BaseKieMarketImageNode):
    MODEL = "ideogram/character"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 360.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The character in a Parisian cafe, drinking coffee at sunset.",
                }),
                "reference_images": ("IMAGE", {
                    "tooltip": "Character reference image(s). Batch for multi-ref (1-5).",
                }),
                "aspect_ratio": (_IDEOGRAM_RATIOS, {"default": "1:1"}),
                "rendering_speed": (_IDEOGRAM_SPEEDS, {"default": "BALANCED"}),
            },
            "optional": {
                "style": (_IDEOGRAM_STYLES, {"default": "AUTO"}),
                "expand_prompt": ("BOOLEAN", {"default": True}),
                "num_images": (_IDEOGRAM_NUM_IMAGES, {"default": "1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        refs = _upload_batch(kwargs.get("reference_images"))
        if not refs:
            raise ValueError("Ideogram Character requires at least one reference image.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "reference_image_urls": refs,
            "aspect_ratio": kwargs["aspect_ratio"],
            "rendering_speed": kwargs["rendering_speed"],
            "style": kwargs.get("style", "AUTO"),
            "expand_prompt": bool(kwargs.get("expand_prompt", True)),
            "num_images": str(kwargs.get("num_images", "1")),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class IdeogramCharacterEdit(BaseKieMarketImageNode):
    MODEL = "ideogram/character-edit"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 360.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A fabulous look, head tilted down with a smile.",
                }),
                "image": ("IMAGE", {"tooltip": "Target image."}),
                "mask": ("IMAGE", {"tooltip": "Mask (white = edit, black = preserve)."}),
                "reference_images": ("IMAGE", {
                    "tooltip": "Character reference image(s). Batch for multi-ref (1-5).",
                }),
                "rendering_speed": (_IDEOGRAM_SPEEDS, {"default": "BALANCED"}),
            },
            "optional": {
                "style": (_IDEOGRAM_STYLES, {"default": "AUTO"}),
                "expand_prompt": ("BOOLEAN", {"default": True}),
                "num_images": (_IDEOGRAM_NUM_IMAGES, {"default": "1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img_url = _upload_first(kwargs.get("image"))
        mask_url = _upload_first(kwargs.get("mask"))
        refs = _upload_batch(kwargs.get("reference_images"))
        if not refs:
            raise ValueError("Ideogram Character Edit requires at least one reference.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img_url,
            "mask_url": mask_url,
            "reference_image_urls": refs,
            "rendering_speed": kwargs["rendering_speed"],
            "style": kwargs.get("style", "AUTO"),
            "expand_prompt": bool(kwargs.get("expand_prompt", True)),
            "num_images": str(kwargs.get("num_images", "1")),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class IdeogramCharacterRemix(BaseKieMarketImageNode):
    MODEL = "ideogram/character-remix"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 360.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The same character in a fantasy forest scene at twilight.",
                }),
                "image": ("IMAGE", {"tooltip": "Source image to remix."}),
                "reference_images": ("IMAGE", {
                    "tooltip": "Character reference image(s). Batch for multi-ref (1-5).",
                }),
                "image_weight": ("INT", {"default": 50, "min": 0, "max": 100, "step": 5}),
                "rendering_speed": (_IDEOGRAM_SPEEDS, {"default": "BALANCED"}),
            },
            "optional": {
                "style": (_IDEOGRAM_STYLES, {"default": "AUTO"}),
                "expand_prompt": ("BOOLEAN", {"default": True}),
                "num_images": (_IDEOGRAM_NUM_IMAGES, {"default": "1"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img_url = _upload_first(kwargs.get("image"))
        refs = _upload_batch(kwargs.get("reference_images"))
        if not refs:
            raise ValueError("Ideogram Character Remix requires at least one reference.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img_url,
            "reference_image_urls": refs,
            "image_weight": int(kwargs["image_weight"]),
            "rendering_speed": kwargs["rendering_speed"],
            "style": kwargs.get("style", "AUTO"),
            "expand_prompt": bool(kwargs.get("expand_prompt", True)),
            "num_images": str(kwargs.get("num_images", "1")),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieIdeogramV3T2I": IdeogramV3T2I,
    "GenesisKieIdeogramV3Edit": IdeogramV3Edit,
    "GenesisKieIdeogramV3Remix": IdeogramV3Remix,
    "GenesisKieIdeogramCharacter": IdeogramCharacter,
    "GenesisKieIdeogramCharacterEdit": IdeogramCharacterEdit,
    "GenesisKieIdeogramCharacterRemix": IdeogramCharacterRemix,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieIdeogramV3T2I": "Ideogram V3 (T2I)",
    "GenesisKieIdeogramV3Edit": "Ideogram V3 Edit",
    "GenesisKieIdeogramV3Remix": "Ideogram V3 Remix",
    "GenesisKieIdeogramCharacter": "Ideogram Character",
    "GenesisKieIdeogramCharacterEdit": "Ideogram Character Edit",
    "GenesisKieIdeogramCharacterRemix": "Ideogram Character Remix",
}
