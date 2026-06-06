"""OpenAI Sora 2 video generation nodes (via Kie.ai).

Covers the 4 core Sora 2 endpoints in Kie.ai's Market:

- Sora 2 Text-to-Video         (sora-2-text-to-video)
- Sora 2 Image-to-Video        (sora-2-image-to-video)
- Sora 2 Pro Text-to-Video     (sora-2-pro-text-to-video)
- Sora 2 Pro Image-to-Video    (sora-2-pro-image-to-video)

Pricing per Kie.ai (June 2026):
- sora-2 std: $0.15 per 10s, 720p, watermark optional
- sora-2-pro std: $0.45 per 10s, 720p
- sora-2-pro HD: $1.00 per 10s, 1080p

Note: Sora 2 API is planned to be discontinued by OpenAI on Sep 24, 2026.
Plan migration paths to Veo 3.1 / Wan 2.7 before that date.

Parameter schemas extracted verbatim from Kie.ai docs cURL examples.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


# Sora 2 uses non-standard aspect_ratio values: "landscape" / "portrait" / "square"
_SORA_RATIOS = ["landscape", "portrait", "square"]


class _SoraBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Sora 2 t2v/i2v (std + pro tiers).

    Sora 2 uses:
    - ``aspect_ratio``: "landscape" / "portrait" / "square" (not numerical)
    - ``n_frames``: string "10" (10s clip)
    - ``remove_watermark``: bool (Kie can remove the OpenAI watermark)
    - ``character_id_list``: array of pre-trained character IDs (max 5)
    - ``upload_method``: "s3" (for image-to-video)
    """

    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A professor enthusiastically lecturing.",
                }),
                "aspect_ratio": (_SORA_RATIOS, {"default": "landscape"}),
                "n_frames": (["10"], {"default": "10"}),
            },
            "optional": {
                "remove_watermark": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove the OpenAI Sora watermark.",
                }),
                "character_ids": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated character IDs (max 5).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "n_frames": str(kwargs["n_frames"]),
            "remove_watermark": bool(kwargs.get("remove_watermark", True)),
        }
        ids = self._csv((kwargs.get("character_ids") or "").strip())
        if ids:
            if len(ids) > 5:
                raise ValueError(f"Sora 2: max 5 character IDs, got {len(ids)}.")
            body["character_id_list"] = ids
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


class _SoraI2VMixin:
    """Adds image_urls + upload_method + progressCallBackUrl for i2v variants."""

    def build_input(self, **kwargs: Any) -> dict[str, Any]:  # type: ignore[override]
        body = super().build_input(**kwargs)  # type: ignore[misc]
        imgs = self._csv((kwargs.get("image_urls") or "").strip())  # type: ignore[attr-defined]
        if not imgs:
            raise ValueError("Sora 2 I2V requires at least one image_url.")
        body["image_urls"] = imgs
        body["upload_method"] = "s3"
        return body


class Sora2T2V(_SoraBase):
    """OpenAI Sora 2 text-to-video. Standard tier, 720p, $0.15/10s."""
    MODEL = "sora-2-text-to-video"


class Sora2I2V(_SoraI2VMixin, _SoraBase):
    """OpenAI Sora 2 image-to-video. Standard tier."""
    MODEL = "sora-2-image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        base = super().INPUT_TYPES()
        base["required"]["image_urls"] = ("STRING", {
            "default": "",
            "tooltip": "Comma-separated image URLs (at least one required).",
        })
        return base


class Sora2ProT2V(_SoraBase):
    """OpenAI Sora 2 Pro text-to-video. Pro tier, $0.45/10s std, $1.00/10s HD."""
    MODEL = "sora-2-pro-text-to-video"


class Sora2ProI2V(_SoraI2VMixin, _SoraBase):
    """OpenAI Sora 2 Pro image-to-video. Pro tier."""
    MODEL = "sora-2-pro-image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        base = super().INPUT_TYPES()
        base["required"]["image_urls"] = ("STRING", {
            "default": "",
            "tooltip": "Comma-separated image URLs (at least one required).",
        })
        return base


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSora2T2V": Sora2T2V,
    "GenesisKieSora2I2V": Sora2I2V,
    "GenesisKieSora2ProT2V": Sora2ProT2V,
    "GenesisKieSora2ProI2V": Sora2ProI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSora2T2V": "Kie — Sora 2 (T2V)",
    "GenesisKieSora2I2V": "Kie — Sora 2 (I2V)",
    "GenesisKieSora2ProT2V": "Kie — Sora 2 Pro (T2V)",
    "GenesisKieSora2ProI2V": "Kie — Sora 2 Pro (I2V)",
}
