"""OpenAI 4o Image API node (via Kie.ai's DEDICATED endpoint).

The 4o Image API is OpenAI's legacy GPT-4o multimodal image generation.
Unlike the newer ``gpt-image-2-*`` Market models, the 4o Image API uses
a dedicated endpoint pattern (like Veo / Runway).

Endpoints:
- POST /api/v1/gpt4o-image/generate     (camelCase body, no input wrapper)
- GET  /api/v1/gpt4o-image/record-info  (polling by taskId)

Status uses ``successFlag`` (Veo-style: 0=gen, 1=success, 2/3=failed).
Result URLs live at ``data.response.resultUrls``.

Body schema per docs.kie.ai cURL:
- prompt (required): text description
- filesUrl (optional array): input images for image-edit mode
- size: "1:1" / "3:2" / "2:3"
- isEnhance: bool — auto-enhance the prompt
- uploadCn: bool — use China-region upload for asset
- enableFallback: bool — fall back to alternative model on failure
- fallbackModel: e.g. "FLUX_MAX" if enableFallback=true
- nVariants: 1, 2, or 4 (per Kie marketing page)

Cost: starts at $0.03 per call (per kie.ai/4o-image-api product page).
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKie4oImageNode


_4O_SIZES = ["1:1", "3:2", "2:3"]
_4O_FALLBACK_MODELS = ["FLUX_MAX", "FLUX_PRO", "NONE"]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


class GPT4oImage(BaseKie4oImageNode):
    """OpenAI 4o Image — multi-variant image generation (dedicated API).

    Supports text-to-image (no filesUrl) and image-edit (with filesUrl
    array containing 1+ source images).
    """

    # MODEL field is unused for 4o (no Kie ``model`` field in body), but we
    # set it for logging consistency with the rest of the catalog.
    MODEL = "gpt4o-image"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 600.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A photorealistic sunset over the mountains.",
                }),
                "size": (_4O_SIZES, {"default": "1:1"}),
                "n_variants": ("INT", {
                    "default": 1, "min": 1, "max": 4,
                    "tooltip": "1, 2, or 4 (typical). Higher = batch generation.",
                }),
            },
            "optional": {
                "files_url": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated source image URLs (for image-edit mode).",
                }),
                "is_enhance": ("BOOLEAN", {"default": False}),
                "upload_cn": ("BOOLEAN", {"default": False}),
                "enable_fallback": ("BOOLEAN", {"default": False}),
                "fallback_model": (_4O_FALLBACK_MODELS, {"default": "NONE"}),
            },
        }

    def build_4o_request(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI kwargs into KieClient.run_4o_image() kwargs."""
        request: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "size": kwargs["size"],
            "n_variants": int(kwargs["n_variants"]),
            "is_enhance": bool(kwargs.get("is_enhance", False)),
            "upload_cn": bool(kwargs.get("upload_cn", False)),
            "enable_fallback": bool(kwargs.get("enable_fallback", False)),
        }

        files = _csv((kwargs.get("files_url") or "").strip())
        if files:
            request["files_url"] = files

        fallback = kwargs.get("fallback_model", "NONE")
        if request["enable_fallback"] and fallback and fallback != "NONE":
            request["fallback_model"] = fallback

        return request


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGPT4oImage": GPT4oImage,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGPT4oImage": "Kie — GPT 4o Image (dedicated)",
}
