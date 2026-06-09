"""Video utility nodes: Topaz upscale, Infinitalk lip-sync, Grok upscale/extend."""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor, upload_video_frames, upload_audio


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


class TopazVideoUpscale(BaseKieMarketVideoNode):
    """Topaz Video Upscale. Input as IMAGE batch (frames)."""

    MODEL = "topaz/video-upscale"
    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1800.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Source video as IMAGE batch (N frames)."}),
                "upscale_factor": (["1", "2", "4"], {"default": "2"}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        video_tensor = kwargs.get("video")
        if video_tensor is None:
            raise ValueError("Topaz Video Upscale requires video input (IMAGE batch).")
        fps = int(kwargs.get("fps", 24))
        return {
            "video_url": upload_video_frames(video_tensor, fps=fps),
            "upscale_factor": str(kwargs["upscale_factor"]),
        }


class InfinitalkFromAudio(BaseKieMarketVideoNode):
    """Infinitalk — lip-sync video from portrait IMAGE + AUDIO."""

    MODEL = "infinitalk/from-audio"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Portrait image."}),
                "audio": ("AUDIO", {"tooltip": "Audio (will sync mouth to it)."}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A young woman with long dark hair talking on a podcast.",
                }),
                "resolution": (["480p", "720p"], {"default": "480p"}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = kwargs.get("image")
        aud = kwargs.get("audio")
        if img is None:
            raise ValueError("Infinitalk requires image input.")
        if aud is None:
            raise ValueError("Infinitalk requires audio input.")

        body: dict[str, Any] = {
            "image_url": _upload_first(img),
            "audio_url": upload_audio(aud),
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
        }
        seed = int(kwargs.get("seed") or 0)
        if seed >= 10000:
            body["seed"] = seed
        return body


class GrokImagineUpscale(BaseKieMarketVideoNode):
    """Grok Imagine Video Upscale (task_id-only)."""

    MODEL = "grok-imagine/upscale"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {"default": "", "tooltip": "Task ID from a prior Kie video task."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        tid = (kwargs.get("task_id") or "").strip()
        if not tid:
            raise ValueError("Grok Imagine Upscale requires task_id.")
        if len(tid) > 100:
            raise ValueError(f"task_id exceeds 100 chars (got {len(tid)}).")
        return {"task_id": tid}


class GrokImagineExtend(BaseKieMarketVideoNode):
    """Grok Imagine Video Extend (task_id-only)."""

    MODEL = "grok-imagine/extend"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {"default": ""}),
                "extend_times": (["6"], {"default": "6"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "extend_at": ("INT", {"default": 0, "min": 0, "max": 60}),
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


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieTopazVideoUpscale": TopazVideoUpscale,
    "GenesisKieInfinitalkFromAudio": InfinitalkFromAudio,
    "GenesisKieGrokImagineUpscale": GrokImagineUpscale,
    "GenesisKieGrokImagineExtend": GrokImagineExtend,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieTopazVideoUpscale": "Topaz Video Upscale",
    "GenesisKieInfinitalkFromAudio": "Infinitalk (Audio → Video)",
    "GenesisKieGrokImagineUpscale": "Grok Imagine Upscale",
    "GenesisKieGrokImagineExtend": "Grok Imagine Extend",
}
