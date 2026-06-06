"""Runway Gen-4 Turbo + Aleph video generation nodes (via Kie.ai).

Covers the 3 Runway endpoints in Kie.ai:

- Runway Gen-4 Turbo (text-to-video / image-to-video)
- Runway Gen-4 Turbo Video Extend
- Runway Aleph (video-to-video editing)

**This module uses Kie.ai's DEDICATED Runway endpoints**, not the Market
`createTask` endpoint. The key differences:

- POST /api/v1/runway/generate  (camelCase body, no ``input`` wrapper)
- POST /api/v1/runway/extend
- POST /api/v1/aleph/generate
- GET  /api/v1/runway/record-detail?taskId=X   (polling)

Response shape: ``data.videoInfo.videoUrl`` (not ``data.resultJson``).

The KieClient exposes ``run_runway(path, body)`` for this pattern.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..base import BaseKieNode, CATEGORY_VIDEO
from ...client import KieClient, KieError, download_to_output

log = logging.getLogger("genesis_kie")


def _runway_first_url(data: dict[str, Any]) -> str | None:
    """Extract the videoUrl from a Runway record-detail response.

    Shape: ``data.videoInfo.videoUrl``.
    """
    video_info = data.get("videoInfo") or {}
    if isinstance(video_info, dict):
        url = video_info.get("videoUrl")
        if isinstance(url, str) and url:
            return url
    # Fallback: some early implementations may put videoUrl at top level.
    direct = data.get("videoUrl")
    if isinstance(direct, str) and direct:
        return direct
    return None


# ============================================================ Base

class _BaseRunwayNode(BaseKieNode):
    """Common scaffolding for Runway dedicated-API video nodes."""

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"

    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1200.0

    # Subclasses set this.
    RUNWAY_PATH: ClassVar[str] = ""

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
        """Build the (camelCase, no ``input`` wrapper) Runway request body."""
        raise NotImplementedError

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.RUNWAY_PATH:
            raise KieError(f"{type(self).__name__} did not declare RUNWAY_PATH.")

        body = self.build_runway_body(**kwargs)
        log.debug("Runway node=%s path=%s body=%s", type(self).__name__, self.RUNWAY_PATH, body)

        with KieClient() as client:
            last_state = [None]

            def on_progress(d: dict[str, Any]) -> None:
                state = d.get("state")
                if state != last_state[0]:
                    log.info("Runway task state=%s", state)
                    last_state[0] = state

            data = client.run_runway(
                self.RUNWAY_PATH,
                body,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
                progress_callback=on_progress,
            )

        url = _runway_first_url(data)
        if not url:
            raise KieError(
                "Runway returned no videoUrl. "
                f"data keys: {list(data.keys())}, "
                f"videoInfo: {data.get('videoInfo')}"
            )
        path = download_to_output(url, prefix="kie_runway", fallback_ext="mp4")
        return (path,)


# ============================================================ Generate

class RunwayGenerate(_BaseRunwayNode):
    """Runway Gen-4 Turbo — text-to-video or image-to-video.

    Per docs:
    - Required: prompt
    - Optional: imageUrl (turns t2v into i2v), duration ("5"/"10"),
      quality ("720p"/"1080p"), aspectRatio ("16:9"/"9:16"), waterMark
    - Note: imageUrl supports JPG/PNG, max 10MB
    - Quality "1080p" only compatible with duration "5"
    """

    RUNWAY_PATH = "/api/v1/runway/generate"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A fluffy orange cat dancing in a colorful room with disco lights.",
                }),
                "duration": (["5", "10"], {"default": "5"}),
                "quality": (["720p", "1080p"], {
                    "default": "720p",
                    "tooltip": "1080p only works with duration=5 per docs.",
                }),
                "aspectRatio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "imageUrl": ("STRING", {
                    "default": "",
                    "tooltip": "Optional reference image URL. When set, becomes image-to-video.",
                }),
                "waterMark": ("STRING", {
                    "default": "",
                    "tooltip": "Optional watermark text overlay.",
                }),
            },
        }

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
        duration = str(kwargs["duration"])
        quality = kwargs["quality"]
        if quality == "1080p" and duration != "5":
            raise ValueError(
                "Runway: quality=1080p is only compatible with duration=5 (per docs)."
            )

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": duration,
            "quality": quality,
            "aspectRatio": kwargs["aspectRatio"],
        }
        img = (kwargs.get("imageUrl") or "").strip()
        if img:
            body["imageUrl"] = img
        wm = (kwargs.get("waterMark") or "").strip()
        if wm:
            body["waterMark"] = wm
        return body


# ============================================================ Extend

class RunwayExtend(_BaseRunwayNode):
    """Runway Gen-4 Turbo — video extension.

    Extends a previously-generated Runway video. Per docs:
    - Required: taskId (from a previous Runway generation)
    - Optional: prompt (describes how to extend), quality, waterMark
    - Note: Videos generated at 1080p CANNOT be extended (per docs).
    """

    RUNWAY_PATH = "/api/v1/runway/extend"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "taskId": ("STRING", {
                    "default": "",
                    "tooltip": "Task ID from a prior Runway generate call (must be 720p).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "The scene continues with more energy and motion.",
                }),
                "quality": (["720p"], {"default": "720p"}),
            },
            "optional": {
                "waterMark": ("STRING", {"default": ""}),
            },
        }

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
        tid = (kwargs.get("taskId") or "").strip()
        if not tid:
            raise ValueError("Runway Extend requires taskId from a prior generation.")
        body: dict[str, Any] = {
            "taskId": tid,
            "prompt": kwargs["prompt"],
            "quality": kwargs["quality"],
        }
        wm = (kwargs.get("waterMark") or "").strip()
        if wm:
            body["waterMark"] = wm
        return body


# ============================================================ Aleph

class RunwayAleph(_BaseRunwayNode):
    """Runway Aleph — text-guided video-to-video transformation.

    Per docs:
    - Required: prompt + videoUrl (the input video to transform)
    - Optional: waterMark, uploadCn (bool, defaults to false)
    - Outputs are capped at 5 seconds (input >5s only processes first 5s)

    Aleph supports: object addition/removal, relighting, style changes,
    new camera angles — while preserving motion and timing of source clip.
    """

    RUNWAY_PATH = "/api/v1/aleph/generate"
    TIMEOUT_SECONDS = 1500.0  # Aleph can be slower than standard generate.

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A majestic eagle soaring through mountain clouds at sunset.",
                }),
                "videoUrl": ("STRING", {
                    "default": "",
                    "tooltip": "Input video URL to transform (max 5s processed).",
                }),
            },
            "optional": {
                "waterMark": ("STRING", {"default": ""}),
                "uploadCn": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "True = use China-region upload (better for CN access).",
                }),
            },
        }

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("videoUrl") or "").strip()
        if not v:
            raise ValueError("Runway Aleph requires videoUrl.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "videoUrl": v,
            "uploadCn": bool(kwargs.get("uploadCn", False)),
        }
        wm = (kwargs.get("waterMark") or "").strip()
        if wm:
            body["waterMark"] = wm
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieRunwayGenerate": RunwayGenerate,
    "GenesisKieRunwayExtend": RunwayExtend,
    "GenesisKieRunwayAleph": RunwayAleph,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieRunwayGenerate": "Kie — Runway Gen-4 Turbo",
    "GenesisKieRunwayExtend": "Kie — Runway Extend",
    "GenesisKieRunwayAleph": "Kie — Runway Aleph (V2V)",
}
