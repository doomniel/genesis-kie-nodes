"""Wan 2.7 Image generation nodes (via Kie.ai).

Alibaba Wan 2.7 unified image model — supports text-to-image and image
editing in a single endpoint. The Pro tier uses the same schema with
better fidelity.

Covers the 2 Wan Image endpoints:

- wan/2-7-image       (standard tier)
- wan/2-7-image-pro   (pro tier)

Per docs.kie.ai cURL — both endpoints share an identical body schema:
- prompt: text instruction
- input_urls: optional reference images (when present, becomes I2I/edit)
- n: number of images to generate (1-4)
- enable_sequential: generate as a sequence (for storyboards/sets)
- resolution: "2K" (only documented value)
- thinking_mode: enable reasoning step for complex prompts
- watermark: include Kie watermark in output
- seed: deterministic seed (0 = random)
- bbox_list: bounding boxes for region-specific edits (per input image)
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketImageNode


_WAN_RESOLUTIONS = ["2K"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


class _WanImageBase(BaseKieMarketImageNode):
    """Shared scaffolding for Wan 2.7 Image (standard + pro).

    Both tiers use identical params; subclasses only set ``MODEL``.
    """

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
                "n": ("INT", {
                    "default": 1, "min": 1, "max": 4,
                    "tooltip": "Number of images to generate (1-4).",
                }),
                "resolution": (_WAN_RESOLUTIONS, {"default": "2K"}),
            },
            "optional": {
                "input_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (optional, makes I2I).",
                }),
                "enable_sequential": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate images as a coherent sequence (e.g. for storyboards).",
                }),
                "thinking_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable reasoning step for complex prompts (higher cost).",
                }),
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
        imgs = _csv((kwargs.get("input_urls") or "").strip())
        if imgs:
            body["input_urls"] = imgs
        return body


class Wan27Image(_WanImageBase):
    """Wan 2.7 Image (standard tier — text-to-image and image edit unified)."""
    MODEL = "wan/2-7-image"


class Wan27ImagePro(_WanImageBase):
    """Wan 2.7 Image Pro (premium tier — higher fidelity)."""
    MODEL = "wan/2-7-image-pro"


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieWan27Image": Wan27Image,
    "GenesisKieWan27ImagePro": Wan27ImagePro,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieWan27Image": "Kie — Wan 2.7 Image",
    "GenesisKieWan27ImagePro": "Kie — Wan 2.7 Image Pro",
}
