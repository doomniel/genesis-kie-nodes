"""OpenAI Sora 2 video generation nodes (via GenesisLab proxy / Kie.ai).

8 nodes:
- Sora 2 (T2V + I2V std)
- Sora 2 Pro (T2V + I2V, with size: standard/high)
- Sora Watermark Remover (Sora.com URL → external, NOT refactored as IMAGE)
- Sora 2 Characters (create character from video clip)
- Sora 2 Characters Pro (from task_id + timestamps — no media inputs)
- Sora 2 Pro Storyboard (multi-scene with optional ref images)

Note: character_ids in standard nodes are IDs (not URLs), so they stay as STRING.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor, upload_video_frames


_SORA_RATIOS = ["landscape", "portrait", "square"]


def _upload_batch(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


def _upload_batch_optional(image_tensor: Any) -> list[str]:
    if image_tensor is None:
        return []
    return _upload_batch(image_tensor)


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Sora 2 std

class _SoraStdBase(BaseKieMarketVideoNode):
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
                "remove_watermark": ("BOOLEAN", {"default": True}),
                "character_ids": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated character IDs (max 5). From Sora 2 Characters node.",
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
        ids = _csv((kwargs.get("character_ids") or "").strip())
        if ids:
            if len(ids) > 5:
                raise ValueError(f"Sora 2: max 5 character IDs, got {len(ids)}.")
            body["character_id_list"] = ids
        return body


class Sora2T2V(_SoraStdBase):
    MODEL = "sora-2-text-to-video"


class Sora2I2V(_SoraStdBase):
    MODEL = "sora-2-image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        base = super().INPUT_TYPES()
        base["required"]["images"] = ("IMAGE", {"tooltip": "Reference image(s). Batch for multi-ref."})
        return base

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("Sora 2 I2V requires at least one image.")
        body["image_urls"] = urls
        body["upload_method"] = "s3"
        return body


# ============================================================ Sora 2 Pro

class _SoraProBase(BaseKieMarketVideoNode):
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
                "size": (["standard", "high"], {"default": "standard"}),
            },
            "optional": {
                "remove_watermark": ("BOOLEAN", {"default": True}),
                "character_ids": ("STRING", {"default": "", "tooltip": "Comma-separated character IDs (max 5)."}),
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
        ids = _csv((kwargs.get("character_ids") or "").strip())
        if ids:
            if len(ids) > 5:
                raise ValueError(f"Sora 2 Pro: max 5 character IDs, got {len(ids)}.")
            body["character_id_list"] = ids
        return body


class Sora2ProT2V(_SoraProBase):
    MODEL = "sora-2-pro-text-to-video"


class Sora2ProI2V(_SoraProBase):
    MODEL = "sora-2-pro-image-to-video"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        base = super().INPUT_TYPES()
        base["required"]["images"] = ("IMAGE", {"tooltip": "Reference image(s). Batch for multi-ref."})
        return base

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("Sora 2 Pro I2V requires at least one image.")
        body["image_urls"] = urls
        return body


# ============================================================ Watermark Remover

class SoraWatermarkRemover(BaseKieMarketVideoNode):
    """Sora Watermark Remover — clean OpenAI Sora.com watermarks.

    REQUIRES an external Sora.com URL (sora.chatgpt.com). NOT refactored
    to AUDIO/IMAGE input since the source must be a hosted Sora URL.
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
                    "tooltip": "Sora.com URL (must start with sora.chatgpt.com).",
                }),
            },
            "optional": {
                "upload_method": (["s3", "oss"], {"default": "s3"}),
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
    """Sora 2 Characters — create reusable character from a video clip."""

    MODEL = "sora-2-characters"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "character_video": ("IMAGE", {"tooltip": "Character video as IMAGE batch."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
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
        v = kwargs.get("character_video")
        if v is None:
            raise ValueError("Sora 2 Characters requires character_video.")
        fps = int(kwargs.get("fps", 24))
        return {
            "character_file_url": [upload_video_frames(v, fps=fps)],
            "character_prompt": kwargs["character_prompt"],
            "safety_instruction": kwargs["safety_instruction"],
        }


class Sora2CharactersPro(BaseKieMarketVideoNode):
    """Sora 2 Characters Pro — extract character from an existing task (task_id-only)."""

    MODEL = "sora-2-characters-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "origin_task_id": ("STRING", {"default": "", "tooltip": "Task ID from a previous Sora 2 generation."}),
                "timestamps": ("STRING", {"default": "3.55,5.55", "tooltip": "Comma-separated start,end seconds."}),
                "character_user_name": ("STRING", {"default": "my_character_01"}),
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
            raise ValueError("Sora 2 Characters Pro requires timestamps.")
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

    Scenes from script (duration:prompt per line). Optional reference
    images (batch) — one per scene in order.
    """

    MODEL = "sora-2-pro-storyboard"
    POLL_INTERVAL_SECONDS = 8.0
    TIMEOUT_SECONDS = 2400.0

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
                    "tooltip": "<duration>: <prompt> per line. Total <= 25s.",
                }),
                "aspect_ratio": (_SORA_RATIOS, {"default": "landscape"}),
                "size": (["standard", "high"], {"default": "standard"}),
            },
            "optional": {
                "remove_watermark": ("BOOLEAN", {"default": True}),
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s), one per scene in order."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        scenes = self._parse_scenes(kwargs["scenes_script"])
        if not scenes:
            raise ValueError("Sora 2 Pro Storyboard requires at least one scene.")
        total = sum(s["duration"] for s in scenes)
        if total > 25:
            raise ValueError(f"Storyboard: scenes total {total}s, max 25s.")

        refs = _upload_batch_optional(kwargs.get("reference_images"))
        if refs:
            for idx, ref_url in enumerate(refs):
                if idx < len(scenes):
                    scenes[idx]["reference_image_url"] = ref_url

        return {
            "scenes": scenes,
            "aspect_ratio": kwargs["aspect_ratio"],
            "size": kwargs["size"],
            "remove_watermark": bool(kwargs.get("remove_watermark", True)),
        }

    @staticmethod
    def _parse_scenes(script: str) -> list[dict[str, Any]]:
        scenes: list[dict[str, Any]] = []
        for line in script.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
                raise ValueError(f"Storyboard line missing ':' separator: {line!r}.")
            dur_str, prompt = line.split(":", 1)
            try:
                duration = int(dur_str.strip())
            except ValueError as exc:
                raise ValueError(f"Invalid duration in line {line!r}: {exc}") from exc
            if not (1 <= duration <= 25):
                raise ValueError(f"Scene duration {duration}s out of range (1-25).")
            scenes.append({"duration": duration, "prompt": prompt.strip()})
        return scenes


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
    "GenesisKieSora2T2V": "Sora 2 (T2V)",
    "GenesisKieSora2I2V": "Sora 2 (I2V)",
    "GenesisKieSora2ProT2V": "Sora 2 Pro (T2V)",
    "GenesisKieSora2ProI2V": "Sora 2 Pro (I2V)",
    "GenesisKieSoraWatermarkRemover": "Sora Watermark Remover",
    "GenesisKieSora2Characters": "Sora 2 Characters (create)",
    "GenesisKieSora2CharactersPro": "Sora 2 Characters Pro (from task)",
    "GenesisKieSora2ProStoryboard": "Sora 2 Pro Storyboard (multi-scene)",
}
