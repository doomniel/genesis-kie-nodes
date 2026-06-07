"""Image utility nodes: Topaz Upscale + Recraft Background/Upscale.

Covers 3 utility endpoints:

- topaz/image-upscale          (AI upscaling 2x/4x with Topaz)
- recraft/remove-background    (transparent PNG background removal)
- recraft/crisp-upscale        (Recraft's crisp upscaler, integrates Topaz tech)

These nodes are for post-processing existing images (no T2I generation).
Per docs.kie.ai cURL: minimal API — only image URL + optional scale.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode


_TOPAZ_FACTORS = ["2", "4"]  # Per docs: "2" or "4" string values


class TopazImageUpscale(BaseKieMarketImageNode):
    """Topaz AI image upscaler (2x or 4x, premium quality).

    Per docs cURL: ``image_url`` + ``upscale_factor`` (string "2" or "4").
    """

    MODEL = "topaz/image-upscale"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 600.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Source image URL to upscale.",
                }),
                "upscale_factor": (_TOPAZ_FACTORS, {
                    "default": "2",
                    "tooltip": "Upscale factor: 2x (faster, cheaper) or 4x (max quality).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Topaz Image Upscale requires image_url.")

        return {
            "image_url": img,
            "upscale_factor": str(kwargs["upscale_factor"]),
        }


class RecraftRemoveBackground(BaseKieMarketImageNode):
    """Recraft AI background removal — transparent PNG output.

    Per docs cURL: ``image`` (singular, NOT image_url).
    Supports PNG / JPG / WEBP. Max 5MB, 16MP, 4096px max, 256px min.
    """

    MODEL = "recraft/remove-background"
    POLL_INTERVAL_SECONDS = 2.0
    TIMEOUT_SECONDS = 180.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("STRING", {
                    "default": "",
                    "tooltip": "Source image URL (max 5MB, max 16MP, 256-4096px per side).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image") or "").strip()
        if not img:
            raise ValueError("Recraft Remove Background requires image URL.")

        return {"image": img}


class RecraftCrispUpscale(BaseKieMarketImageNode):
    """Recraft Crisp Upscale — sharper image upscaling (Topaz-integrated tech).

    Per docs cURL: ``image`` (singular).
    """

    MODEL = "recraft/crisp-upscale"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 480.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("STRING", {
                    "default": "",
                    "tooltip": "Source image URL to upscale crisply.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image") or "").strip()
        if not img:
            raise ValueError("Recraft Crisp Upscale requires image URL.")

        return {"image": img}


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieTopazImageUpscale": TopazImageUpscale,
    "GenesisKieRecraftRemoveBackground": RecraftRemoveBackground,
    "GenesisKieRecraftCrispUpscale": RecraftCrispUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieTopazImageUpscale": "Kie — Topaz Image Upscale",
    "GenesisKieRecraftRemoveBackground": "Kie — Recraft Remove Background",
    "GenesisKieRecraftCrispUpscale": "Kie — Recraft Crisp Upscale",
}
