"""Black Forest Labs Flux Kontext node (via GenesisLab proxy / Kie.ai dedicated endpoint).

Flux Kontext is BFL's image generation + editing model. Unlike the Market
``flux-2/*`` family, Kontext uses a dedicated endpoint pattern.

Accepts an optional IMAGE input. When connected, the node operates in
edit mode (image_edit). When omitted, it operates as pure T2I.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieFluxKontextNode
from ...client.upload import upload_image_tensor


_KONTEXT_MODELS = ["flux-kontext-pro", "flux-kontext-max"]
_KONTEXT_RATIOS = ["preserve", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
_KONTEXT_FORMATS = ["jpeg", "png"]


def _upload_first_optional(image_tensor: Any) -> str | None:
    if image_tensor is None:
        return None
    return upload_image_tensor(image_tensor[0:1])


class FluxKontextPro(BaseKieFluxKontextNode):
    """Flux Kontext Pro/Max — T2I + image-edit (dedicated API).

    Use ``model`` parameter to switch between Pro and Max tiers.
    """

    MODEL = "flux-kontext-pro"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 480.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A serene mountain landscape at sunset with a lake "
                        "reflecting the orange sky."
                    ),
                }),
                "model": (_KONTEXT_MODELS, {
                    "default": "flux-kontext-pro",
                    "tooltip": "Pro = balanced. Max = maximum quality (higher cost).",
                }),
                "aspect_ratio": (_KONTEXT_RATIOS, {
                    "default": "preserve",
                    "tooltip": "'preserve' keeps source ratio (only meaningful in edit mode).",
                }),
                "output_format": (_KONTEXT_FORMATS, {"default": "jpeg"}),
            },
            "optional": {
                "input_image": ("IMAGE", {
                    "tooltip": "Optional source image. Connected = edit mode, omitted = T2I.",
                }),
                "prompt_upsampling": ("BOOLEAN", {"default": False}),
                "enable_translation": ("BOOLEAN", {"default": True}),
                "safety_tolerance": ("INT", {"default": 2, "min": 0, "max": 6}),
            },
        }

    def build_kontext_request(self, **kwargs: Any) -> dict[str, Any]:
        request: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "output_format": kwargs["output_format"],
            "prompt_upsampling": bool(kwargs.get("prompt_upsampling", False)),
            "enable_translation": bool(kwargs.get("enable_translation", True)),
            "safety_tolerance": int(kwargs.get("safety_tolerance", 2)),
        }

        ratio = kwargs.get("aspect_ratio", "preserve")
        if ratio and ratio != "preserve":
            request["aspect_ratio"] = ratio

        img_url = _upload_first_optional(kwargs.get("input_image"))
        if img_url:
            request["input_image"] = img_url

        return request


NODE_CLASS_MAPPINGS: dict[str, type] = {"GenesisKieFluxKontextPro": FluxKontextPro}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieFluxKontextPro": "Flux Kontext (Pro/Max, dedicated)",
}
