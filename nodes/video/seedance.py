"""ByteDance Seedance video generation nodes.

This batch covers the two most-used Seedance tiers:

- **Seedance 2.0** — full-quality, supports up to 1080p
- **Seedance 2.0 Fast** — cheaper, up to 720p

Both use the Market endpoint ``/api/v1/jobs/createTask``.

Seedance is unique in that one node handles ALL the generation modes:
text-to-video, image-to-video (first frame, first+last), and multimodal
reference-to-video. The mode is determined by which optional inputs are
provided.

Pricing reference (Kie.ai, 2026):
    Seedance 2.0 480p no video    $0.095/s
    Seedance 2.0 720p no video    $0.205/s
    Seedance 2.0 1080p no video   $0.51/s
    Seedance 2.0 Fast 720p        $0.165/s
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]


class _SeedanceBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Seedance 2.0 tiers.

    Per docs, Seedance supports these mutually-exclusive modes (set via
    different optional inputs):

    - First frame only (image-to-video)
    - First + last frame (transition video)
    - Multimodal reference (images / videos / audio as references)

    If none of these are set, defaults to text-to-video.
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    # Tiers override this with the supported resolutions for their endpoint.
    SUPPORTED_RESOLUTIONS: ClassVar[list[str]] = ["480p", "720p", "1080p"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A serene beach at sunset with waves crashing.",
                }),
                "resolution": (cls.SUPPORTED_RESOLUTIONS, {"default": "720p"}),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                }),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "first_frame_url": ("STRING", {
                    "default": "",
                    "tooltip": "Image URL for first frame (image-to-video mode).",
                }),
                "last_frame_url": ("STRING", {
                    "default": "",
                    "tooltip": "Image URL for last frame "
                               "(transition video — requires first_frame_url).",
                }),
                "reference_image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs "
                               "(multimodal reference mode).",
                }),
                "reference_video_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference video URLs.",
                }),
                "reference_audio_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference audio URLs.",
                }),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "web_search": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
            "generate_audio": bool(kwargs.get("generate_audio", False)),
            "return_last_frame": bool(kwargs.get("return_last_frame", False)),
            "web_search": bool(kwargs.get("web_search", False)),
        }

        first = (kwargs.get("first_frame_url") or "").strip()
        last = (kwargs.get("last_frame_url") or "").strip()
        ref_imgs = self._csv((kwargs.get("reference_image_urls") or "").strip())
        ref_vids = self._csv((kwargs.get("reference_video_urls") or "").strip())
        ref_auds = self._csv((kwargs.get("reference_audio_urls") or "").strip())

        # Validate mutual exclusivity per Seedance docs.
        has_first_last = bool(first or last)
        has_refs = bool(ref_imgs or ref_vids or ref_auds)
        if has_first_last and has_refs:
            raise ValueError(
                "Seedance: first/last frame mode and reference mode are "
                "mutually exclusive."
            )

        if first:
            body["first_frame_url"] = first
        if last:
            body["last_frame_url"] = last
        if ref_imgs:
            body["reference_image_urls"] = ref_imgs
        if ref_vids:
            body["reference_video_urls"] = ref_vids
        if ref_auds:
            body["reference_audio_urls"] = ref_auds

        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


class Seedance20(_SeedanceBase):
    """ByteDance Seedance 2.0 — full quality, up to 1080p."""

    MODEL = "bytedance/seedance-2"
    SUPPORTED_RESOLUTIONS = ["480p", "720p", "1080p"]


class Seedance20Fast(_SeedanceBase):
    """ByteDance Seedance 2.0 Fast — cheaper, up to 720p."""

    MODEL = "bytedance/seedance-2-fast"
    SUPPORTED_RESOLUTIONS = ["480p", "720p"]


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSeedance20": Seedance20,
    "GenesisKieSeedance20Fast": Seedance20Fast,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSeedance20": "Kie — Seedance 2.0",
    "GenesisKieSeedance20Fast": "Kie — Seedance 2.0 Fast",
}
