"""HappyHorse 1.0 video generation nodes (complete family).

HappyHorse-1.0 by Alibaba ATH — ranked #1 on Artificial Analysis Text-to-Video
leaderboard in April 2026. Features joint audio-video generation in a single
forward pass (15B-param unified self-attention Transformer).

Covers all 4 HappyHorse endpoints in Kie.ai:

- HappyHorse Text-to-Video        (happyhorse/text-to-video)
- HappyHorse Image-to-Video       (happyhorse/image-to-video)
- HappyHorse Reference-to-Video   (happyhorse/reference-to-video)  — 1-9 ref images
- HappyHorse Video Edit           (happyhorse/video-edit)          — V2V editing

All use the Market endpoint /api/v1/jobs/createTask.

Parameter schemas inferred from:
- Kie.ai's HappyHorse product page (kie.ai/happyhorse-1-0)
- Runware HappyHorse-1.0 docs (mirrors the same model)
- APIXO HappyHorse API docs (mirrors the same model)

Common parameters across modes:
- prompt (required) — max 5000 non-Chinese / 2500 Chinese chars
- resolution: "720P" | "1080P" (default 1080P)
- duration: 3-15 seconds (default 5)  [text/image/reference-to-video only]
- aspect_ratio: "16:9" (default) | "9:16" | "1:1" | "4:3" | "3:4"
                [text-to-video + reference-to-video only; I2V follows uploaded
                 first frame; Edit follows input video]
- seed: 0-2147483647
- audio_setting: "auto" | "origin"  [video-edit only]

NOTE: Param naming follows the snake_case Kie convention (resolution, not Resolution).
The 720P/1080P enum values are UPPERCASE per docs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


_HAPPYHORSE_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
_HAPPYHORSE_RESOLUTIONS = ["720P", "1080P"]


# ============================================================ Text-to-Video

class HappyHorseT2V(BaseKieMarketVideoNode):
    """HappyHorse 1.0 text-to-video.

    Generate a video purely from a text description. Native audio generation
    is supported (joint audio-video model).
    """

    MODEL = "happyhorse/text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A silver horse runs through a rain-soaked neon alley, cinematic camera move.",
                }),
                "resolution": (_HAPPYHORSE_RESOLUTIONS, {"default": "1080P"}),
                "aspect_ratio": (_HAPPYHORSE_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ============================================================ Image-to-Video

class HappyHorseI2V(BaseKieMarketVideoNode):
    """HappyHorse 1.0 image-to-video.

    Animate a static image into a short video. Dimensions follow the
    uploaded first frame; only resolution tier is configurable.
    """

    MODEL = "happyhorse/image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "First-frame image URL (required, max 10MB).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Cinematic motion, subtle camera move.",
                }),
                "resolution": (_HAPPYHORSE_RESOLUTIONS, {"default": "1080P"}),
                "duration": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("HappyHorse I2V requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ============================================================ Reference-to-Video

class HappyHorseR2V(BaseKieMarketVideoNode):
    """HappyHorse 1.0 reference-to-video.

    Generate a video guided by 1-9 reference images. Per docs:
    - Use ``character1``, ``character2``, ... in the prompt to refer to
      the corresponding images (order matches the array).
    - Min resolution per image: short side >= 400px
    - 720p+ clear images are recommended; avoid blurry / compressed sources.
    """

    MODEL = "happyhorse/reference-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated 1-9 reference image URLs. Use 'character1', 'character2' in prompt to reference by order.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "character1 walks toward the camera in a cinematic outdoor scene.",
                }),
                "resolution": (_HAPPYHORSE_RESOLUTIONS, {"default": "1080P"}),
                "aspect_ratio": (_HAPPYHORSE_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {
                    "default": 5, "min": 3, "max": 15, "step": 1,
                }),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = self._csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("HappyHorse R2V requires at least one reference image.")
        if len(imgs) > 9:
            raise ValueError(f"HappyHorse R2V: max 9 reference images, got {len(imgs)}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_urls": imgs,
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Video Edit

class HappyHorseEdit(BaseKieMarketVideoNode):
    """HappyHorse 1.0 video edit (V2V).

    Refine, adjust, or extend existing video content via prompt. Per docs:
    - video_url is required (source video, max 10MB)
    - Up to 5 optional reference images (each 300px+ per side; AR 1:2.5–2.5:1)
    - Audio handling: ``audio_setting`` = ``"auto"`` (model decides) or
      ``"origin"`` (keep source audio when supported)
    - Output dimensions inherited from input video; only resolution tier
      is selectable
    """

    MODEL = "happyhorse/video-edit"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Source video URL (required, max 10MB).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Apply cinematic color grading and a slow camera push-in.",
                }),
                "resolution": (_HAPPYHORSE_RESOLUTIONS, {"default": "1080P"}),
                "audio_setting": (["auto", "origin"], {
                    "default": "auto",
                    "tooltip": "auto = model decides; origin = keep source audio.",
                }),
            },
            "optional": {
                "reference_image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated optional reference images (0-5).",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        if not v:
            raise ValueError("HappyHorse Edit requires video_url.")
        refs = self._csv((kwargs.get("reference_image_urls") or "").strip())
        if len(refs) > 5:
            raise ValueError(f"HappyHorse Edit: max 5 reference images, got {len(refs)}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "video_url": v,
            "resolution": kwargs["resolution"],
            "audio_setting": kwargs.get("audio_setting", "auto"),
        }
        if refs:
            body["image_urls"] = refs
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieHappyHorseT2V": HappyHorseT2V,
    "GenesisKieHappyHorseI2V": HappyHorseI2V,
    "GenesisKieHappyHorseR2V": HappyHorseR2V,
    "GenesisKieHappyHorseEdit": HappyHorseEdit,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieHappyHorseT2V": "Kie — HappyHorse 1.0 (T2V)",
    "GenesisKieHappyHorseI2V": "Kie — HappyHorse 1.0 (I2V)",
    "GenesisKieHappyHorseR2V": "Kie — HappyHorse 1.0 Reference-to-Video",
    "GenesisKieHappyHorseEdit": "Kie — HappyHorse 1.0 Video Edit",
}
