"""Black Forest Labs Flux-2 image nodes (via Kie.ai).

Flux-2 is BFL's flagship image model (released Nov 2025), with enhanced
realism, crisper text rendering, and native editing capabilities.

Covers the 4 Flux-2 endpoints in Kie.ai's Market:

- flux-2/pro-text-to-image    (Pro tier, T2I)
- flux-2/pro-image-to-image   (Pro tier, I2I)
- flux-2/flex-text-to-image   (Flex tier, T2I)
- flux-2/flex-image-to-image  (Flex tier, I2I)

Per docs.kie.ai cURL: minimal required body is prompt + aspect_ratio +
resolution. The Flex tier offers more flexibility/cost optimization;
the Pro tier targets production-quality output.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode


_FLUX2_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
_FLUX2_RESOLUTIONS = ["1K", "2K"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


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
    """Shared scaffolding for Flux-2 image-to-image (Pro + Flex)."""

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
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (1-10 for multi-ref).",
                }),
                "aspect_ratio": (_FLUX2_RATIOS, {"default": "1:1"}),
                "resolution": (_FLUX2_RESOLUTIONS, {"default": "1K"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = _csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError(f"{type(self).__name__} requires at least one image_url.")

        return {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
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
    "GenesisKieFlux2ProT2I": "Kie — Flux-2 Pro (T2I)",
    "GenesisKieFlux2ProI2I": "Kie — Flux-2 Pro (I2I)",
    "GenesisKieFlux2FlexT2I": "Kie — Flux-2 Flex (T2I)",
    "GenesisKieFlux2FlexI2I": "Kie — Flux-2 Flex (I2I)",
}
