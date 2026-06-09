"""Black Forest Labs Flux-2 image nodes (via the GenesisLab proxy).

Covers the 4 Flux-2 endpoints:

- flux-2/pro-text-to-image    (Pro tier, T2I)
- flux-2/pro-image-to-image   (Pro tier, I2I)
- flux-2/flex-text-to-image   (Flex tier, T2I)
- flux-2/flex-image-to-image  (Flex tier, I2I)

I2I nodes accept a native ComfyUI IMAGE tensor as input. The tensor is
uploaded to GenesisLab temp storage (R2, 24h TTL) and the resulting
public URL is sent to the upstream API. If a batched tensor is supplied
(N > 1), all images of the batch are uploaded as multi-reference inputs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode
from ...client.upload import upload_image_tensor


_FLUX2_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
_FLUX2_RESOLUTIONS = ["1K", "2K"]


def _upload_batch(image_tensor: Any) -> list[str]:
    """Iterate over the batch dim of a ComfyUI IMAGE tensor and upload
    each frame to GenesisLab temp storage.

    Returns the list of public URLs in batch order.
    """
    if image_tensor is None:
        raise ValueError("image tensor is required")
    if not hasattr(image_tensor, "shape"):
        raise ValueError(f"expected IMAGE tensor, got {type(image_tensor).__name__}")

    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    urls: list[str] = []
    for i in range(n):
        # Slice keeps batch dimension: (1, H, W, C)
        slice_tensor = image_tensor[i:i + 1]
        urls.append(upload_image_tensor(slice_tensor))
    return urls


# ============================================================ Common bases

class _FluxT2IBase(BaseKieMarketImageNode):
    """Shared scaffolding for Flux-2 text-to-image (Pro + Flex)."""

    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 300.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A humanoid figure with a vintage television for a head, "
                        "wearing a yellow raincoat in an outdoor setting."
                    ),
                }),
                "aspect_ratio": (_FLUX2_RATIOS, {"default": "1:1"}),
                "resolution": (_FLUX2_RESOLUTIONS, {"default": "1K"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
        }


class _FluxI2IBase(BaseKieMarketImageNode):
    """Shared scaffolding for Flux-2 image-to-image (Pro + Flex).

    Accepts a ComfyUI IMAGE tensor (possibly batched). Each image in the
    batch is uploaded to GenesisLab temp storage and the resulting public
    URLs are sent to the upstream API as the multi-reference input.
    """

    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 400.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Transform the subject into a Renaissance painting style.",
                }),
                "image": ("IMAGE", {
                    "tooltip": (
                        "Reference image. Connect a Load Image node, or a "
                        "batched tensor for multi-reference editing (up to 10)."
                    ),
                }),
                "aspect_ratio": (_FLUX2_RATIOS, {"default": "1:1"}),
                "resolution": (_FLUX2_RESOLUTIONS, {"default": "1K"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("image"))
        if not urls:
            raise ValueError(f"{type(self).__name__}: no images to upload")

        return {
            "prompt": kwargs["prompt"],
            # API spec key is still "image_urls" (list of URLs)
            "input_urls": urls,
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
        }


# ============================================================ Concrete nodes

class Flux2ProT2I(_FluxT2IBase):
    """Flux-2 Pro Text-to-Image (production-quality)."""
    MODEL = "flux-2/pro-text-to-image"


class Flux2ProI2I(_FluxI2IBase):
    """Flux-2 Pro Image-to-Image (production-quality, multi-reference editing)."""
    MODEL = "flux-2/pro-image-to-image"


class Flux2FlexT2I(_FluxT2IBase):
    """Flux-2 Flex Text-to-Image (cost-optimized tier)."""
    MODEL = "flux-2/flex-text-to-image"


class Flux2FlexI2I(_FluxI2IBase):
    """Flux-2 Flex Image-to-Image (cost-optimized editing tier)."""
    MODEL = "flux-2/flex-image-to-image"


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieFlux2ProT2I": Flux2ProT2I,
    "GenesisKieFlux2ProI2I": Flux2ProI2I,
    "GenesisKieFlux2FlexT2I": Flux2FlexT2I,
    "GenesisKieFlux2FlexI2I": Flux2FlexI2I,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieFlux2ProT2I": "Flux-2 Pro (T2I)",
    "GenesisKieFlux2ProI2I": "Flux-2 Pro (I2I)",
    "GenesisKieFlux2FlexT2I": "Flux-2 Flex (T2I)",
    "GenesisKieFlux2FlexI2I": "Flux-2 Flex (I2I)",
}
