"""Image utility nodes: Topaz Upscale + Recraft Background/Upscale.

All 3 nodes take a single IMAGE input.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_TOPAZ_FACTORS = ["2", "4"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


class TopazImageUpscale(BaseKieMarketImageNode):
    """Topaz AI image upscaler (2x or 4x)."""

    MODEL = "topaz/image-upscale"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 600.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to upscale."}),
                "upscale_factor": (_TOPAZ_FACTORS, {"default": "2"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        url = _upload_first(kwargs.get("image"))
        return {
            "image_url": url,
            "upscale_factor": str(kwargs["upscale_factor"]),
        }


class RecraftRemoveBackground(BaseKieMarketImageNode):
    """Recraft AI background removal — transparent PNG output."""

    MODEL = "recraft/remove-background"
    POLL_INTERVAL_SECONDS = 2.0
    TIMEOUT_SECONDS = 180.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image (max 16MP)."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        url = _upload_first(kwargs.get("image"))
        return {"image": url}


class RecraftCrispUpscale(BaseKieMarketImageNode):
    """Recraft Crisp Upscale — sharper image upscaling."""

    MODEL = "recraft/crisp-upscale"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 480.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Source image to upscale."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        url = _upload_first(kwargs.get("image"))
        return {"image": url}


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieTopazImageUpscale": TopazImageUpscale,
    "GenesisKieRecraftRemoveBackground": RecraftRemoveBackground,
    "GenesisKieRecraftCrispUpscale": RecraftCrispUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieTopazImageUpscale": "Topaz Image Upscale",
    "GenesisKieRecraftRemoveBackground": "Recraft Remove Background",
    "GenesisKieRecraftCrispUpscale": "Recraft Crisp Upscale",
}
