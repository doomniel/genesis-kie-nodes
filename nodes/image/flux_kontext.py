"""Black Forest Labs Flux Kontext node (via Kie.ai's DEDICATED endpoint).

Flux Kontext is BFL's image generation + editing model with strong prompt
adherence. Unlike the Market ``flux-2/*`` family, Kontext uses a dedicated
endpoint pattern (like Veo / Runway / 4o Image).

Endpoints:
- POST /api/v1/flux/kontext/generate     (camelCase body, no input wrapper)
- GET  /api/v1/flux/kontext/record-info  (polling by taskId)

Status uses ``successFlag`` (Veo-style: 0=gen, 1=success, 2/3=failed).
Result URLs live at ``data.response.resultUrls``.

Body schema per docs.kie.ai cURL:
- prompt (required): text instruction
- model: "flux-kontext-pro" or "flux-kontext-max"
- aspectRatio (optional): "16:9", "1:1", etc — if omitted, source ratio preserved
- outputFormat: "jpeg" / "png"
- promptUpsampling: bool — auto-enhance the prompt via VLM
- enableTranslation: bool — auto-translate non-English prompts
- safetyTolerance: int 0-6 (per BFL docs, 2 is default)
- inputImage (optional): URL — if provided, edit mode; else T2I

Per fal.ai pricing reference: Kontext [pro] $0.04/image, Kontext [max]
$0.08/image. Kie.ai may have different pricing.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieFluxKontextNode


_KONTEXT_MODELS = ["flux-kontext-pro", "flux-kontext-max"]
_KONTEXT_RATIOS = ["preserve", "1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3", "21:9"]
_KONTEXT_FORMATS = ["jpeg", "png"]


class FluxKontextPro(BaseKieFluxKontextNode):
    """Flux Kontext Pro — text-to-image + image-edit (dedicated API).

    The same node handles both T2I (no ``input_image``) and image-edit
    (with ``input_image`` URL). Use the ``model`` parameter to switch
    between Pro and Max tiers without needing separate node classes.
    """

    MODEL = "flux-kontext-pro"  # Default; can be overridden per-call via param
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
                    "tooltip": "Pro = balanced. Max = maximum performance (higher cost).",
                }),
                "aspect_ratio": (_KONTEXT_RATIOS, {
                    "default": "preserve",
                    "tooltip": "'preserve' keeps source ratio (only meaningful for image-edit).",
                }),
                "output_format": (_KONTEXT_FORMATS, {"default": "jpeg"}),
            },
            "optional": {
                "input_image": ("STRING", {
                    "default": "",
                    "tooltip": "Optional source image URL. Empty = T2I. Filled = image-edit.",
                }),
                "prompt_upsampling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Auto-enhance prompt via VLM before generation.",
                }),
                "enable_translation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Auto-translate non-English prompts.",
                }),
                "safety_tolerance": ("INT", {
                    "default": 2, "min": 0, "max": 6,
                    "tooltip": "0 = strictest filter, 6 = most permissive (default 2).",
                }),
            },
        }

    def build_kontext_request(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI kwargs into KieClient.run_flux_kontext() kwargs."""
        request: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "output_format": kwargs["output_format"],
            "prompt_upsampling": bool(kwargs.get("prompt_upsampling", False)),
            "enable_translation": bool(kwargs.get("enable_translation", True)),
            "safety_tolerance": int(kwargs.get("safety_tolerance", 2)),
        }

        # aspect_ratio: 'preserve' → omit field (keeps source ratio for edit mode).
        ratio = kwargs.get("aspect_ratio", "preserve")
        if ratio and ratio != "preserve":
            request["aspect_ratio"] = ratio

        # input_image: only include if non-empty (switches into edit mode).
        img = (kwargs.get("input_image") or "").strip()
        if img:
            request["input_image"] = img

        return request


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieFluxKontextPro": FluxKontextPro,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieFluxKontextPro": "Kie — Flux Kontext (Pro/Max, dedicated)",
}
