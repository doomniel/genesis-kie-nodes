"""Google image generation nodes (via GenesisLab proxy / Kie.ai).

Covers Imagen 4 family (3 T2I variants) and Nano Banana family
(T2I + Edit + Pro + 2). Body field naming varies:
- NanoBananaEdit uses image_urls (required)
- NanoBananaPro / NanoBanana2 use image_input (optional)
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_IMAGEN_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]
_BANANA_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
_BANANA_RESOLUTIONS = ["1K", "2K", "4K"]
_OUTPUT_FORMATS = ["png", "jpeg", "webp"]


def _upload_batch(image_tensor: Any, required: bool = True) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        if required:
            raise ValueError("image tensor required")
        return []
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class _Imagen4Base(BaseKieMarketImageNode):
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
                "seed": ("STRING", {"default": ""}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "negative_prompt": kwargs.get("negative_prompt", "") or "",
            "aspect_ratio": kwargs["aspect_ratio"],
            "seed": (kwargs.get("seed") or "").strip(),
        }


class Imagen4(_Imagen4Base):
    MODEL = "google/imagen4"


class Imagen4Fast(_Imagen4Base):
    MODEL = "google/imagen4-fast"


class Imagen4Ultra(_Imagen4Base):
    MODEL = "google/imagen4-ultra"


class NanoBanana(BaseKieMarketImageNode):
    """Google Nano Banana T2I (Gemini 2.5 Flash Image)."""

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
    """Google Nano Banana Edit (Gemini 2.5 Flash Image — I2I)."""

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
                "images": ("IMAGE", {
                    "tooltip": "Reference image(s). Batch for multi-ref (max 14).",
                }),
                "aspect_ratio": (_BANANA_RATIOS, {"default": "1:1"}),
            },
            "optional": {
                "output_format": (_OUTPUT_FORMATS, {"default": "png"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"), required=True)
        if not urls:
            raise ValueError("Nano Banana Edit requires at least one image.")
        if len(urls) > 14:
            raise ValueError(f"Nano Banana Edit: max 14 images, got {len(urls)}.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
            "output_format": kwargs.get("output_format", "png"),
        }


class NanoBananaPro(BaseKieMarketImageNode):
    """Google Nano Banana Pro (Gemini 3 Pro Image — 4K capable)."""

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
                "resolution": (_BANANA_RESOLUTIONS, {"default": "1K"}),
            },
            "optional": {
                "image_input": ("IMAGE", {
                    "tooltip": "Optional reference image(s). Batch for multi-ref.",
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
        urls = _upload_batch(kwargs.get("image_input"), required=False)
        body["image_input"] = urls
        return body


class NanoBanana2(BaseKieMarketImageNode):
    """Google Nano Banana 2 (Gemini 3.1 Flash Image)."""

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
                "image_input": ("IMAGE", {
                    "tooltip": "Optional reference image(s). Batch for multi-ref.",
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
        urls = _upload_batch(kwargs.get("image_input"), required=False)
        body["image_input"] = urls
        return body


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
    "GenesisKieImagen4": "Imagen 4",
    "GenesisKieImagen4Fast": "Imagen 4 Fast",
    "GenesisKieImagen4Ultra": "Imagen 4 Ultra",
    "GenesisKieNanoBanana": "Nano Banana (Gemini 2.5 Flash)",
    "GenesisKieNanoBananaEdit": "Nano Banana Edit",
    "GenesisKieNanoBananaPro": "Nano Banana Pro (Gemini 3 Pro)",
    "GenesisKieNanoBanana2": "Nano Banana 2 (Gemini 3.1 Flash)",
}
