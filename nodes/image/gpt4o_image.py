"""OpenAI 4o Image API node (dedicated endpoint via GenesisLab proxy)."""

from __future__ import annotations

from typing import Any

from ..base import BaseKie4oImageNode
from ...client.upload import upload_image_tensor


_4O_SIZES = ["1:1", "3:2", "2:3"]
_4O_FALLBACK_MODELS = ["FLUX_MAX", "FLUX_PRO", "NONE"]


def _upload_batch_optional(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        return []
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


class GPT4oImage(BaseKie4oImageNode):
    """OpenAI 4o Image — T2I + image-edit (dedicated API)."""

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
                "n_variants": ("INT", {"default": 1, "min": 1, "max": 4}),
            },
            "optional": {
                "files_input": ("IMAGE", {
                    "tooltip": "Optional source image(s). Connect a batch for image-edit mode.",
                }),
                "is_enhance": ("BOOLEAN", {"default": False}),
                "upload_cn": ("BOOLEAN", {"default": False}),
                "enable_fallback": ("BOOLEAN", {"default": False}),
                "fallback_model": (_4O_FALLBACK_MODELS, {"default": "NONE"}),
            },
        }

    def build_4o_request(self, **kwargs: Any) -> dict[str, Any]:
        request: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "size": kwargs["size"],
            "n_variants": int(kwargs["n_variants"]),
            "is_enhance": bool(kwargs.get("is_enhance", False)),
            "upload_cn": bool(kwargs.get("upload_cn", False)),
            "enable_fallback": bool(kwargs.get("enable_fallback", False)),
        }

        urls = _upload_batch_optional(kwargs.get("files_input"))
        if urls:
            request["files_url"] = urls

        fallback = kwargs.get("fallback_model", "NONE")
        if request["enable_fallback"] and fallback and fallback != "NONE":
            request["fallback_model"] = fallback

        return request


NODE_CLASS_MAPPINGS: dict[str, type] = {"GenesisKieGPT4oImage": GPT4oImage}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGPT4oImage": "GPT 4o Image (dedicated)",
}
