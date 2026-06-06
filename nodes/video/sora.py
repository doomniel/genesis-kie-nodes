"""OpenAI Sora 2 video generation nodes (via Kie.ai).

Covers all 8 Sora 2 endpoints in Kie.ai's Market:

Core (4):
- sora-2-text-to-video
- sora-2-image-to-video
- sora-2-pro-text-to-video       (with size: standard/high)
- sora-2-pro-image-to-video      (with size: standard/high)

Extras (4):
- sora-watermark-remover         (clean Sora.com watermarked videos)
- sora-2-characters              (create reusable character from a clip)
- sora-2-characters-pro          (create character from a prior task at timestamps)
- sora-2-pro-storyboard          (TODO: spec not fully exposed yet)

Pricing per Kie.ai (June 2026):
- sora-2 std:           $0.15 per 10s, 720p
- sora-2-pro standard:  $0.45 per 10s, 720p
- sora-2-pro high (HD): $1.00 per 10s, 1080p
- watermark remover:    ~$0.05 per video

Note: OpenAI plans to discontinue the Sora 2 API on Sep 24, 2026.
Plan migration paths to Wan 2.7 / Veo 3.1 before that date.

Parameter schemas extracted from Kie.ai docs cURL examples.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


# Sora 2 uses non-standard aspect_ratio values: "landscape" / "portrait" / "square"
_SORA_RATIOS = ["landscape", "portrait", "square"]


# ============================================================ Sora 2 std

class _SoraStdBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Sora 2 std (no ``size`` parameter)."""

    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A professor enthusiastically lecturing in a sunny classroom.",
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
                    "tooltip": "Comma-separated character IDs (max 5). Get via Sora 2 Characters node.",
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


class Sora2T2V(_SoraStdBase):
    """OpenAI Sora 2 text-to-video. Standard tier, 720p, $0.15/10s."""
    MODEL = "sora-2-text-to-video"


class Sora2I2V(_SoraStdBase):
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

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        imgs = self._csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Sora 2 I2V requires at least one image_url.")
        body["image_urls"] = imgs
        body["upload_method"] = "s3"
        return body


# ============================================================ Sora 2 Pro

class _SoraProBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Sora 2 Pro (with ``size`` parameter).

    Sora 2 Pro adds:
    - ``size``: "standard" (720p) or "high" (1080p HD)
    """

    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1500.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A happy dog running through a sun-drenched garden.",
                }),
                "aspect_ratio": (_SORA_RATIOS, {"default": "landscape"}),
                "n_frames": (["10", "15"], {"default": "10"}),
                "size": (["standard", "high"], {
                    "default": "standard",
                    "tooltip": "standard = 720p ($0.45/10s), high = 1080p HD ($1.00/10s).",
                }),
            },
            "optional": {
                "remove_watermark": ("BOOLEAN", {"default": True}),
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
            "size": kwargs["size"],
            "remove_watermark": bool(kwargs.get("remove_watermark", True)),
        }
        ids = self._csv((kwargs.get("character_ids") or "").strip())
        if ids:
            if len(ids) > 5:
                raise ValueError(f"Sora 2 Pro: max 5 character IDs, got {len(ids)}.")
            body["character_id_list"] = ids
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


class Sora2ProT2V(_SoraProBase):
    """OpenAI Sora 2 Pro text-to-video. With size selector (standard/high)."""
    MODEL = "sora-2-pro-text-to-video"


class Sora2ProI2V(_SoraProBase):
    """OpenAI Sora 2 Pro image-to-video. With size selector."""
    MODEL = "sora-2-pro-image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        base = super().INPUT_TYPES()
        base["required"]["image_urls"] = ("STRING", {
            "default": "",
            "tooltip": "Comma-separated image URLs (at least one required).",
        })
        return base

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        imgs = self._csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Sora 2 Pro I2V requires at least one image_url.")
        body["image_urls"] = imgs
        return body


# ============================================================ Watermark Remover

class SoraWatermarkRemover(BaseKieMarketVideoNode):
    """Sora Watermark Remover — clean OpenAI Sora.com watermarks.

    Per docs: input must be a publicly accessible Sora.com URL
    (starts with sora.chatgpt.com). Cost: ~$0.05 per video.

    If the source video is in draft (not published on Sora), the
    watermark stays but no credits are consumed.
    """

    MODEL = "sora-watermark-remover"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 600.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Sora.com video URL (must start with sora.chatgpt.com).",
                }),
            },
            "optional": {
                "upload_method": (["s3", "oss"], {
                    "default": "s3",
                    "tooltip": "s3 = default; oss = Aliyun (better for China access).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        if not v:
            raise ValueError("Sora Watermark Remover requires video_url.")
        return {
            "video_url": v,
            "upload_method": kwargs.get("upload_method", "s3"),
        }


# ============================================================ Characters

class Sora2Characters(BaseKieMarketVideoNode):
    """Sora 2 Characters — create reusable character from a video clip.

    Per docs cURL example:
        {
          "model": "sora-2-characters",
          "input": {
            "character_file_url": ["https://.../clip.mp4"],
            "character_prompt": "A friendly cartoon character with...",
            "safety_instruction": "Family-friendly content only..."
          }
        }

    The returned task's result includes a ``character_id`` that can then
    be used in any sora-2-* call's ``character_id_list``.
    """

    MODEL = "sora-2-characters"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "character_file_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated character video/clip URLs.",
                }),
                "character_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A friendly cartoon character with expressive eyes and fluid movements.",
                }),
                "safety_instruction": ("STRING", {
                    "multiline": True,
                    "default": "Ensure the animation is family-friendly and contains no violent or inappropriate content.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = self._csv((kwargs.get("character_file_urls") or "").strip())
        if not urls:
            raise ValueError("Sora 2 Characters requires at least one character_file_url.")
        return {
            "character_file_url": urls,
            "character_prompt": kwargs["character_prompt"],
            "safety_instruction": kwargs["safety_instruction"],
        }

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


class Sora2CharactersPro(BaseKieMarketVideoNode):
    """Sora 2 Characters Pro — extract character from an existing task.

    Per docs cURL example:
        {
          "model": "sora-2-characters-pro",
          "input": {
            "origin_task_id": "7118f712c1f35c9b8bf2ad1af68ad482",
            "timestamps": "3.55,5.55",
            "character_user_name": "my_character_01",
            "character_prompt": "A friendly cartoon character...",
            "safety_instruction": "Family-friendly content only..."
          }
        }

    Extracts character from the source task at the given timestamps
    (comma-separated seconds, format "start,end").
    """

    MODEL = "sora-2-characters-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "origin_task_id": ("STRING", {
                    "default": "",
                    "tooltip": "Task ID from a previous Sora 2 generation.",
                }),
                "timestamps": ("STRING", {
                    "default": "3.55,5.55",
                    "tooltip": "Comma-separated 'start,end' seconds (e.g. '3.55,5.55').",
                }),
                "character_user_name": ("STRING", {
                    "default": "my_character_01",
                    "tooltip": "Unique character name to register.",
                }),
                "character_prompt": ("STRING", {
                    "multiline": True,
                    "default": "A friendly cartoon character with expressive eyes and fluid movements.",
                }),
                "safety_instruction": ("STRING", {
                    "multiline": True,
                    "default": "Ensure the animation is family-friendly and contains no violent or inappropriate content.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        tid = (kwargs.get("origin_task_id") or "").strip()
        if not tid:
            raise ValueError("Sora 2 Characters Pro requires origin_task_id.")
        timestamps = (kwargs.get("timestamps") or "").strip()
        if not timestamps:
            raise ValueError("Sora 2 Characters Pro requires timestamps (e.g. '3.55,5.55').")

        return {
            "origin_task_id": tid,
            "timestamps": timestamps,
            "character_user_name": kwargs["character_user_name"],
            "character_prompt": kwargs["character_prompt"],
            "safety_instruction": kwargs["safety_instruction"],
        }


# ============================================================ Storyboard

class Sora2ProStoryboard(BaseKieMarketVideoNode):
    """Sora 2 Pro Storyboard — multi-scene cinematic video (up to 25s).

    Per Kie.ai's product page (kie.ai/es/sora-2-pro-storyboard):
    - Up to 25-second multi-scene videos
    - Each scene has its own prompt + optional reference image
    - Smooth transitions between scenes
    - Maintains visual consistency across the timeline

    **WARNING**: The exact request body shape was not exposed in the docs
    HTML at the time of writing — only the model name was confirmed. The
    body schema below uses an inferred ``scenes`` array based on the
    feature description, matching the patterns used by other Sora-2-pro
    endpoints. If the actual API rejects this shape, check docs.kie.ai
    for the canonical OpenAPI spec and adjust ``build_input`` accordingly.

    Tracked at: <https://docs.kie.ai/market/sora2/sora-2-pro-storyboard>
    """

    MODEL = "sora-2-pro-storyboard"
    POLL_INTERVAL_SECONDS = 8.0
    TIMEOUT_SECONDS = 2400.0  # Multi-scene takes longer.

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "scenes_script": ("STRING", {
                    "multiline": True,
                    "default": (
                        "5: A wide establishing shot of a futuristic city at dawn.\n"
                        "5: The camera pushes through neon-lit streets toward a tower.\n"
                        "5: Inside, a person looks out the window as light catches their face."
                    ),
                    "tooltip": (
                        "One scene per line as '<duration>: <prompt>'. "
                        "Durations in seconds; total cannot exceed 25s. "
                        "Format inferred from Kie product page."
                    ),
                }),
                "aspect_ratio": (_SORA_RATIOS, {"default": "landscape"}),
                "size": (["standard", "high"], {"default": "standard"}),
            },
            "optional": {
                "remove_watermark": ("BOOLEAN", {"default": True}),
                "reference_image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (one per scene, in order).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        scenes = self._parse_scenes(kwargs["scenes_script"])
        if not scenes:
            raise ValueError("Sora 2 Pro Storyboard requires at least one scene.")
        total = sum(s["duration"] for s in scenes)
        if total > 25:
            raise ValueError(f"Sora 2 Pro Storyboard: scenes total {total}s, max 25s allowed.")

        refs = self._csv((kwargs.get("reference_image_urls") or "").strip())
        if refs:
            for idx, ref in enumerate(refs):
                if idx < len(scenes):
                    scenes[idx]["reference_image_url"] = ref

        body: dict[str, Any] = {
            "scenes": scenes,
            "aspect_ratio": kwargs["aspect_ratio"],
            "size": kwargs["size"],
            "remove_watermark": bool(kwargs.get("remove_watermark", True)),
        }
        return body

    @staticmethod
    def _parse_scenes(script: str) -> list[dict[str, Any]]:
        """Parse 'duration: prompt' lines into scene dicts."""
        scenes: list[dict[str, Any]] = []
        for line in script.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(
                    f"Storyboard line missing ':' separator: {line!r}. "
                    "Use '<duration>: <prompt>' format."
                )
            dur_str, prompt = line.split(":", 1)
            try:
                duration = int(dur_str.strip())
            except ValueError as exc:
                raise ValueError(
                    f"Invalid duration in line {line!r}: {exc}"
                ) from exc
            if not (1 <= duration <= 25):
                raise ValueError(
                    f"Scene duration {duration}s out of range (1-25 per scene)."
                )
            scenes.append({"duration": duration, "prompt": prompt.strip()})
        return scenes

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSora2T2V": Sora2T2V,
    "GenesisKieSora2I2V": Sora2I2V,
    "GenesisKieSora2ProT2V": Sora2ProT2V,
    "GenesisKieSora2ProI2V": Sora2ProI2V,
    "GenesisKieSoraWatermarkRemover": SoraWatermarkRemover,
    "GenesisKieSora2Characters": Sora2Characters,
    "GenesisKieSora2CharactersPro": Sora2CharactersPro,
    "GenesisKieSora2ProStoryboard": Sora2ProStoryboard,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSora2T2V": "Kie — Sora 2 (T2V)",
    "GenesisKieSora2I2V": "Kie — Sora 2 (I2V)",
    "GenesisKieSora2ProT2V": "Kie — Sora 2 Pro (T2V)",
    "GenesisKieSora2ProI2V": "Kie — Sora 2 Pro (I2V)",
    "GenesisKieSoraWatermarkRemover": "Kie — Sora Watermark Remover",
    "GenesisKieSora2Characters": "Kie — Sora 2 Characters (create)",
    "GenesisKieSora2CharactersPro": "Kie — Sora 2 Characters Pro (from task)",
    "GenesisKieSora2ProStoryboard": "Kie — Sora 2 Pro Storyboard (multi-scene)",
}
