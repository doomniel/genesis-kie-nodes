"""Google image generation nodes (via Kie.ai).

Covers the 7 Google image endpoints in Kie.ai:

- google/imagen4              (Imagen 4 standard)
- google/imagen4-fast         (Imagen 4 fast tier)
- google/imagen4-ultra        (Imagen 4 ultra/highest quality)
- google/nano-banana          (Gemini 2.5 Flash Image — T2I)
- google/nano-banana-edit     (Gemini 2.5 Flash Image — I2I/edit)
- nano-banana-pro             (Gemini 3 Pro Image — 4K capable)
- nano-banana-2               (Gemini 3.1 Flash Image — newer fast tier)

Parameter schemas extracted from docs.kie.ai cURL examples.

Notes on parameter conventions:
- Imagen 4 family: prompt, negative_prompt, aspect_ratio, seed (string)
- Nano Banana T2I: prompt, aspect_ratio, output_format
- Nano Banana Edit: prompt, image_urls + above
- Nano Banana Pro: prompt, image_input[], aspect_ratio, resolution (1K/2K/4K),
  output_format (png/jpeg)
- Nano Banana 2: similar to Pro with added Gemini 3.1 capabilities
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode


_IMAGEN_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
_BANANA_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
_BANANA_RESOLUTIONS = ["1K", "2K", "4K"]
_OUTPUT_FORMATS = ["png", "jpeg", "webp"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Imagen 4 family

class _Imagen4Base(BaseKieMarketImageNode):
    """Shared scaffolding for Imagen 4 family (standard/fast/ultra)."""

    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A lively comic scene where two colleagues are discussing "
                        "Imagen 4 in a sunny office."
                    ),
                }),
                "aspect_ratio": (_IMAGEN_RATIOS, {"default": "1:1"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("STRING", {
                    "default": "",
                    "tooltip": "Optional seed (Kie expects empty string '' or numeric string).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "negative_prompt": kwargs.get("negative_prompt", "") or "",
            "aspect_ratio": kwargs["aspect_ratio"],
            "seed": (kwargs.get("seed") or "").strip(),  # Kie expects string
        }


class Imagen4(_Imagen4Base):
    """Google Imagen 4 (standard tier)."""
    MODEL = "google/imagen4"


class Imagen4Fast(_Imagen4Base):
    """Google Imagen 4 Fast (fastest tier, lower cost)."""
    MODEL = "google/imagen4-fast"


class Imagen4Ultra(_Imagen4Base):
    """Google Imagen 4 Ultra (highest fidelity, best for production)."""
    MODEL = "google/imagen4-ultra"


# ============================================================ Nano Banana family

class NanoBanana(BaseKieMarketImageNode):
    """Google Nano Banana T2I (Gemini 2.5 Flash Image).

    Fast image generation with hyper-realistic, physics-aware visuals
    and seamless style transformations.
    """

    MODEL = "google/nano-banana"
    POLL_INTERVAL_SECONDS = 2.5
    TIMEOUT_SECONDS = 240.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A photorealistic astronaut riding a horse on Mars at sunset.",
                }),
                "aspect_ratio": (_BANANA_RATIOS, {"default": "1:1"}),
            },
            "optional": {
                "output_format": (_OUTPUT_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "output_format": kwargs.get("output_format", "png"),
        }


class NanoBananaEdit(BaseKieMarketImageNode):
    """Google Nano Banana Edit (Gemini 2.5 Flash Image — I2I).

    Edit existing images with text prompts. Supports multi-image input
    (up to 14 reference images per upstream Gemini docs).
    """

    MODEL = "google/nano-banana-edit"
    POLL_INTERVAL_SECONDS = 2.5
    TIMEOUT_SECONDS = 240.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Change the background to a sunny beach.",
                }),
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (1-14).",
                }),
                "aspect_ratio": (_BANANA_RATIOS, {"default": "1:1"}),
            },
            "optional": {
                "output_format": (_OUTPUT_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Nano Banana Edit requires at least one image_url.")
        if len(imgs) > 14:
            raise ValueError(f"Nano Banana Edit: max 14 images, got {len(imgs)}.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
            "aspect_ratio": kwargs["aspect_ratio"],
            "output_format": kwargs.get("output_format", "png"),
        }


class NanoBananaPro(BaseKieMarketImageNode):
    """Google Nano Banana Pro (Gemini 3 Pro Image — 4K capable).

    Per docs cURL: model is ``nano-banana-pro`` (no google/ prefix).
    Premier tier — sharper 2K-4K imagery, improved text rendering,
    enhanced character consistency.
    """

    MODEL = "nano-banana-pro"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 600.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A multi-panel comic poster with multilingual text rendering.",
                }),
                "aspect_ratio": (_BANANA_RATIOS, {"default": "1:1"}),
                "resolution": (_BANANA_RESOLUTIONS, {
                    "default": "1K",
                    "tooltip": "2K costs 1.5x, 4K costs 2x standard rate.",
                }),
            },
            "optional": {
                "image_input": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (optional, for image-to-image).",
                }),
                "output_format": (_OUTPUT_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "output_format": kwargs.get("output_format", "png"),
        }
        imgs = _csv((kwargs.get("image_input") or "").strip())
        # Kie docs show image_input as array (empty allowed for T2I mode).
        body["image_input"] = imgs
        return body


class NanoBanana2(BaseKieMarketImageNode):
    """Google Nano Banana 2 (Gemini 3.1 Flash Image — fast tier of Pro lineup).

    Per kie.ai marketing: Pro-level quality with Flash-level speed,
    accurate text rendering, strong character consistency.
    """

    MODEL = "google/nano-banana-2"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic product shot of a luxury watch with crisp text branding.",
                }),
                "aspect_ratio": (_BANANA_RATIOS, {"default": "1:1"}),
                "resolution": (_BANANA_RESOLUTIONS, {"default": "1K"}),
            },
            "optional": {
                "image_input": ("STRING", {
                    "default": "",
                    "tooltip": "Optional reference image URLs (comma-separated).",
                }),
                "output_format": (_OUTPUT_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "output_format": kwargs.get("output_format", "png"),
        }
        imgs = _csv((kwargs.get("image_input") or "").strip())
        body["image_input"] = imgs
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieImagen4": Imagen4,
    "GenesisKieImagen4Fast": Imagen4Fast,
    "GenesisKieImagen4Ultra": Imagen4Ultra,
    "GenesisKieNanoBanana": NanoBanana,
    "GenesisKieNanoBananaEdit": NanoBananaEdit,
    "GenesisKieNanoBananaPro": NanoBananaPro,
    "GenesisKieNanoBanana2": NanoBanana2,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieImagen4": "Kie — Imagen 4",
    "GenesisKieImagen4Fast": "Kie — Imagen 4 Fast",
    "GenesisKieImagen4Ultra": "Kie — Imagen 4 Ultra",
    "GenesisKieNanoBanana": "Kie — Nano Banana (Gemini 2.5 Flash)",
    "GenesisKieNanoBananaEdit": "Kie — Nano Banana Edit",
    "GenesisKieNanoBananaPro": "Kie — Nano Banana Pro (Gemini 3 Pro)",
    "GenesisKieNanoBanana2": "Kie — Nano Banana 2 (Gemini 3.1 Flash)",
}
