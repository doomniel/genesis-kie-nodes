"""Alibaba Z-Image generation nodes (via Kie.ai).

Z-Image is Alibaba Tongyi-MAI's photorealistic image model. It supports
bilingual text rendering (English + Chinese) and high-fidelity product
shots. The Turbo variant is a few-step distilled model (8 steps default).

Covers the single Z-image endpoint:

- z-image    (model name; Kie auto-selects variant)

Per docs.kie.ai cURL example — minimal API: prompt, aspect_ratio,
nsfw_checker. Additional params (negative_prompt, seed, num_inference_steps)
inferred from upstream Alibaba/fal.ai patterns and supported by some Kie
catalog variants.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketImageNode


_Z_RATIOS = ["1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"]


class ZImage(BaseKieMarketImageNode):
    """Z-Image text-to-image (Alibaba Tongyi-MAI photorealistic model).

    Per docs cURL: prompt, aspect_ratio, nsfw_checker.
    Note: Z-Image-Turbo does NOT support classifier-free guidance, so
    negative prompts have no effect on the Turbo variant. Place all
    constraints in the positive prompt.
    """

    MODEL = "z-image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "Photorealistic image of a cafe terrace in the Marais "
                        "district of Paris on a Wednesday morning in March 2025."
                    ),
                }),
                "aspect_ratio": (_Z_RATIOS, {"default": "1:1"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", True)),
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieZImage": ZImage,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieZImage": "Z-Image (Alibaba)",
}
