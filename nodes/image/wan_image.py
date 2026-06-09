"""Wan 2.7 Image generation nodes (via GenesisLab proxy / Kie.ai).

Alibaba Wan 2.7 unified image model — supports T2I and I2I in a single endpoint.
Accepts an optional IMAGE input (possibly batched) for I2I/edit mode.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_WAN_RESOLUTIONS = ["2K"]


def _upload_batch_optional(image_tensor: Any) -> list[str]:
    if image_tensor is None:
        return []
    if not hasattr(image_tensor, "shape"):
        return []
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class _WanImageBase(BaseKieMarketImageNode):
    """Shared scaffolding for Wan 2.7 Image (standard + pro)."""

    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 600.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Transform the food photo by replacing the marked "
                        "ingredients with sliced red chili pieces."
                    ),
                }),
                "n": ("INT", {"default": 1, "min": 1, "max": 4}),
                "resolution": (_WAN_RESOLUTIONS, {"default": "2K"}),
            },
            "optional": {
                "input_images": ("IMAGE", {
                    "tooltip": "Optional reference image(s). Connect a batch for multi-ref.",
                }),
                "enable_sequential": ("BOOLEAN", {"default": False}),
                "thinking_mode": ("BOOLEAN", {"default": False}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "n": int(kwargs["n"]),
            "resolution": kwargs["resolution"],
            "enable_sequential": bool(kwargs.get("enable_sequential", False)),
            "thinking_mode": bool(kwargs.get("thinking_mode", False)),
            "watermark": bool(kwargs.get("watermark", False)),
            "seed": int(kwargs.get("seed") or 0),
        }
        urls = _upload_batch_optional(kwargs.get("input_images"))
        if urls:
            body["input_urls"] = urls
        return body


class Wan27Image(_WanImageBase):
    MODEL = "wan/2-7-image"


class Wan27ImagePro(_WanImageBase):
    MODEL = "wan/2-7-image-pro"


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieWan27Image": Wan27Image,
    "GenesisKieWan27ImagePro": Wan27ImagePro,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieWan27Image": "Wan 2.7 Image",
    "GenesisKieWan27ImagePro": "Wan 2.7 Image Pro",
}
