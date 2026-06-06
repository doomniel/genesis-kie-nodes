"""Video utility nodes: Topaz upscale, Infinitalk lip-sync, Grok upscale/extend.

These don't fit cleanly under a single brand family.

- Topaz Video Upscale     (topaz/video-upscale)
- Infinitalk From Audio   (infinitalk/from-audio)
- Grok Imagine Upscale    (grok-imagine/upscale)
- Grok Imagine Extend     (grok-imagine/extend)

The Grok upscale/extend take a ``task_id`` from a previous Kie generation
(not an external video URL) — only Kie-generated videos can be processed.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketVideoNode


# ============================================================ Topaz

class TopazVideoUpscale(BaseKieMarketVideoNode):
    """Topaz Video Upscale (AI super-resolution).

    Upscale factor: "1" / "2" / "4". Default 2x.
    Input: MP4/MOV/MKV, max 10MB.
    """

    MODEL = "topaz/video-upscale"
    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1800.0  # Upscaling can be slow.

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Source video URL (MP4/MOV/MKV, max 10MB).",
                }),
                "upscale_factor": (["1", "2", "4"], {"default": "2"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        if not v:
            raise ValueError("Topaz Video Upscale requires video_url.")
        return {
            "video_url": v,
            "upscale_factor": str(kwargs["upscale_factor"]),
        }


# ============================================================ Infinitalk

class InfinitalkFromAudio(BaseKieMarketVideoNode):
    """Infinitalk — lip-sync video from portrait image + audio.

    Required: image_url + audio_url + prompt.
    Resolution: 480p / 720p. Seed range: 10000-1000000.
    """

    MODEL = "infinitalk/from-audio"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Portrait image URL (required).",
                }),
                "audio_url": ("STRING", {
                    "default": "",
                    "tooltip": "Audio file URL (max 10MB).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A young woman with long dark hair talking on a podcast.",
                }),
                "resolution": (["480p", "720p"], {"default": "480p"}),
            },
            "optional": {
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 1_000_000,
                    "tooltip": "Seed (10000-1000000). 0 = random.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        aud = (kwargs.get("audio_url") or "").strip()
        if not img:
            raise ValueError("Infinitalk requires image_url.")
        if not aud:
            raise ValueError("Infinitalk requires audio_url.")
        body: dict[str, Any] = {
            "image_url": img,
            "audio_url": aud,
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
        }
        seed = int(kwargs.get("seed") or 0)
        if seed >= 10000:
            body["seed"] = seed
        return body


# ============================================================ Grok extras

class GrokImagineUpscale(BaseKieMarketVideoNode):
    """Grok Imagine Video Upscale.

    Takes a ``task_id`` from a previous Kie-generated video task. Only
    Kie AI–generated videos can be upscaled.
    """

    MODEL = "grok-imagine/upscale"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "Task ID from a previous Kie video task (max 100 chars).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        tid = (kwargs.get("task_id") or "").strip()
        if not tid:
            raise ValueError("Grok Imagine Upscale requires task_id.")
        if len(tid) > 100:
            raise ValueError(f"Grok Imagine Upscale: task_id exceeds 100 chars (got {len(tid)}).")
        return {"task_id": tid}


class GrokImagineExtend(BaseKieMarketVideoNode):
    """Grok Imagine Video Extend.

    Extends a previously generated Kie video by N seconds.
    Required: task_id (from a Kie video task).
    """

    MODEL = "grok-imagine/extend"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "Task ID from a previous Kie video task.",
                }),
                "extend_times": (["6"], {
                    "default": "6",
                    "tooltip": "Seconds to extend (per docs example).",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional prompt to guide the extension.",
                }),
                "extend_at": ("INT", {
                    "default": 0, "min": 0, "max": 60,
                    "tooltip": "Timestamp (in seconds) at which to extend.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        tid = (kwargs.get("task_id") or "").strip()
        if not tid:
            raise ValueError("Grok Imagine Extend requires task_id.")
        return {
            "task_id": tid,
            "prompt": (kwargs.get("prompt") or "").strip(),
            "extend_at": int(kwargs.get("extend_at", 0)),
            "extend_times": str(kwargs["extend_times"]),
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieTopazVideoUpscale": TopazVideoUpscale,
    "GenesisKieInfinitalkFromAudio": InfinitalkFromAudio,
    "GenesisKieGrokImagineUpscale": GrokImagineUpscale,
    "GenesisKieGrokImagineExtend": GrokImagineExtend,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieTopazVideoUpscale": "Kie — Topaz Video Upscale",
    "GenesisKieInfinitalkFromAudio": "Kie — Infinitalk (Audio → Video)",
    "GenesisKieGrokImagineUpscale": "Kie — Grok Imagine Upscale",
    "GenesisKieGrokImagineExtend": "Kie — Grok Imagine Extend",
}
