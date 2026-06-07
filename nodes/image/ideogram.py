"""Ideogram image nodes (via Kie.ai).

Ideogram is the gold standard for AI text rendering in images
(typography, posters, logos with legible text). Created by former Google
Brain researchers (Norouzi, Chan, Saharia, Ho).

Covers all 6 Ideogram endpoints in Kie.ai's Market:

- ideogram-v3/text-to-image     (V3 flagship T2I)
- ideogram-v3/edit              (V3 inpainting / targeted edit)
- ideogram-v3/remix             (V3 transform existing image)
- ideogram/character            (character consistency T2I)
- ideogram/character-edit       (character-preserving edit with mask)
- ideogram/character-remix      (character-preserving remix)

Common parameters across endpoints (per docs.kie.ai cURL examples):
- rendering_speed: "TURBO" / "BALANCED" / "QUALITY"
- style: "AUTO" / "GENERAL" / "REALISTIC" / "DESIGN" / etc
- expand_prompt: bool — let Ideogram enrich the prompt automatically
- num_images: string "1"-"8"
- aspect_ratio: "1:1", "16:9", "4:3", "3:4", "9:16", etc.

V3 endpoints use ``image_size`` (named ratios like landscape_4_3).
Character endpoints use ``aspect_ratio`` (numeric like "1:1").
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode


_IDEOGRAM_SPEEDS = ["TURBO", "BALANCED", "QUALITY"]
_IDEOGRAM_STYLES = ["AUTO", "GENERAL", "REALISTIC", "DESIGN", "RENDER_3D", "ANIME"]
_IDEOGRAM_NUM_IMAGES = ["1", "2", "3", "4", "5", "6", "7", "8"]
_IDEOGRAM_V3_SIZES = [
    "1:1",
    "landscape_4_3", "landscape_3_2", "landscape_16_9",
    "portrait_4_3", "portrait_3_2", "portrait_16_9",
]
_IDEOGRAM_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Ideogram V3

class IdeogramV3T2I(BaseKieMarketImageNode):
    """Ideogram V3 text-to-image (flagship, exceptional text rendering)."""

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
                "rendering_speed": (_IDEOGRAM_SPEEDS, {
                    "default": "BALANCED",
                    "tooltip": "TURBO=$0.03, BALANCED=$0.06, QUALITY=$0.09 (per docs).",
                }),
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
    """Ideogram V3 edit (mask-based inpainting / targeted modifications)."""

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
                "image_url": ("STRING", {"default": "", "tooltip": "Source image URL."}),
                "mask_url": ("STRING", {
                    "default": "",
                    "tooltip": "Mask URL (white = edit, black = preserve).",
                }),
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
        img = (kwargs.get("image_url") or "").strip()
        mask = (kwargs.get("mask_url") or "").strip()
        if not img:
            raise ValueError("Ideogram V3 Edit requires image_url.")
        if not mask:
            raise ValueError("Ideogram V3 Edit requires mask_url.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "mask_url": mask,
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
    """Ideogram V3 remix (transform an existing image with style/content shift)."""

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
                "image_url": ("STRING", {"default": "", "tooltip": "Source image URL."}),
                "image_weight": ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 5,
                    "tooltip": "0-100: how much to preserve source (higher = closer to original).",
                }),
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
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Ideogram V3 Remix requires image_url.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
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


# ============================================================ Ideogram Character

class IdeogramCharacter(BaseKieMarketImageNode):
    """Ideogram Character (one-shot character consistency from reference).

    Best-in-class character consistency: generate multiple images of the
    same character across different scenes/styles using a reference image.
    """

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
                "reference_image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated character reference image URLs (1-5).",
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
        refs = _csv((kwargs.get("reference_image_urls") or "").strip())
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
    """Ideogram Character Edit (mask-based edit preserving character identity).

    Per docs cURL: prompt, image_url, mask_url, reference_image_urls,
    rendering_speed, style, expand_prompt, num_images (string).
    """

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
                "image_url": ("STRING", {"default": "", "tooltip": "Target image URL."}),
                "mask_url": ("STRING", {
                    "default": "",
                    "tooltip": "Mask URL (white = edit region, black = preserve).",
                }),
                "reference_image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Character reference image URLs (comma-separated, 1-5).",
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
        img = (kwargs.get("image_url") or "").strip()
        mask = (kwargs.get("mask_url") or "").strip()
        refs = _csv((kwargs.get("reference_image_urls") or "").strip())
        if not img:
            raise ValueError("Ideogram Character Edit requires image_url.")
        if not mask:
            raise ValueError("Ideogram Character Edit requires mask_url.")
        if not refs:
            raise ValueError("Ideogram Character Edit requires at least one reference image.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "mask_url": mask,
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
    """Ideogram Character Remix (transform with character-preserving constraints)."""

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
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Source image URL to remix.",
                }),
                "reference_image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Character reference URLs (comma-separated, 1-5).",
                }),
                "image_weight": ("INT", {
                    "default": 50, "min": 0, "max": 100, "step": 5,
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
        img = (kwargs.get("image_url") or "").strip()
        refs = _csv((kwargs.get("reference_image_urls") or "").strip())
        if not img:
            raise ValueError("Ideogram Character Remix requires image_url.")
        if not refs:
            raise ValueError("Ideogram Character Remix requires at least one reference image.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
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


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieIdeogramV3T2I": IdeogramV3T2I,
    "GenesisKieIdeogramV3Edit": IdeogramV3Edit,
    "GenesisKieIdeogramV3Remix": IdeogramV3Remix,
    "GenesisKieIdeogramCharacter": IdeogramCharacter,
    "GenesisKieIdeogramCharacterEdit": IdeogramCharacterEdit,
    "GenesisKieIdeogramCharacterRemix": IdeogramCharacterRemix,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieIdeogramV3T2I": "Kie — Ideogram V3 (T2I)",
    "GenesisKieIdeogramV3Edit": "Kie — Ideogram V3 Edit",
    "GenesisKieIdeogramV3Remix": "Kie — Ideogram V3 Remix",
    "GenesisKieIdeogramCharacter": "Kie — Ideogram Character",
    "GenesisKieIdeogramCharacterEdit": "Kie — Ideogram Character Edit",
    "GenesisKieIdeogramCharacterRemix": "Kie — Ideogram Character Remix",
}
