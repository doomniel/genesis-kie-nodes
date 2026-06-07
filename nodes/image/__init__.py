"""Image-generation nodes for Genesis-Kie.

43 image nodes across 11 families:

- Seedream (7):      Bytedance 3.0/4.0/4.5/5.0-Lite × {T2I, Edit/I2I}
- Z-image (1):       Alibaba Tongyi-MAI photorealistic
- Wan Image (2):     Alibaba Wan 2.7 (standard + pro)
- Google (7):        Imagen 4 × 3 + Nano Banana × 4 (T2I/Edit/Pro/v2)
- Flux-2 (4):        BFL Pro/Flex × T2I/I2I
- Grok Imagine (2):  xAI T2I + I2I
- GPT Image (4):     OpenAI 1.5 + 2 × T2I/I2I
- Image Utils (3):   Topaz Upscale + Recraft Remove BG + Crisp Upscale
- Qwen (5):          v1 T2I/I2I/Edit + v2 T2I/Edit
- Ideogram (6):      v3 T2I/Edit/Remix + Character/Edit/Remix
- 4o Image (1):      OpenAI 4o (dedicated endpoint)
- Flux Kontext (1):  BFL Kontext Pro/Max (dedicated endpoint)
"""

from __future__ import annotations

from . import (
    flux2,
    flux_kontext,
    google_image,
    gpt4o_image,
    gpt_image,
    grok_image,
    ideogram,
    image_utils,
    qwen,
    seedream,
    wan_image,
    z_image,
)

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (
    seedream,
    z_image,
    wan_image,
    google_image,
    flux2,
    grok_image,
    gpt_image,
    image_utils,
    qwen,
    ideogram,
    gpt4o_image,
    flux_kontext,
):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
