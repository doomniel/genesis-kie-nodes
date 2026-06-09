"""Runway Gen-4 Turbo + Aleph video nodes (dedicated endpoints, via GenesisLab proxy)."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..base import BaseKieNode, CATEGORY_VIDEO
from ...client import KieClient, KieError, download_to_output
from ...client.upload import upload_image_tensor, upload_video_frames

log = logging.getLogger("genesis_kie")


def _upload_first_optional(image_tensor: Any) -> str | None:
    if image_tensor is None:
        return None
    return upload_image_tensor(image_tensor[0:1])


def _runway_first_url(data: dict[str, Any]) -> str | None:
    video_info = data.get("videoInfo") or {}
    if isinstance(video_info, dict):
        url = video_info.get("videoUrl")
        if isinstance(url, str) and url:
            return url
    direct = data.get("videoUrl")
    if isinstance(direct, str) and direct:
        return direct
    return None


class _BaseRunwayNode(BaseKieNode):
    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "run"

    POLL_INTERVAL_SECONDS = 6.0
    TIMEOUT_SECONDS = 1200.0

    RUNWAY_PATH: ClassVar[str] = ""

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
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
                f"Runway returned no videoUrl. data keys: {list(data.keys())}, "
                f"videoInfo: {data.get('videoInfo')}"
            )
        path = download_to_output(url, prefix="kie_runway", fallback_ext="mp4")
        return (path,)


class RunwayGenerate(_BaseRunwayNode):
    """Runway Gen-4 Turbo — T2V or I2V (when image connected)."""

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
                "quality": (["720p", "1080p"], {"default": "720p"}),
                "aspectRatio": (["16:9", "9:16"], {"default": "16:9"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional reference image (turns T2V into I2V)."}),
                "waterMark": ("STRING", {"default": ""}),
            },
        }

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
        duration = str(kwargs["duration"])
        quality = kwargs["quality"]
        if quality == "1080p" and duration != "5":
            raise ValueError("Runway: quality=1080p only compatible with duration=5.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": duration,
            "quality": quality,
            "aspectRatio": kwargs["aspectRatio"],
        }
        img_url = _upload_first_optional(kwargs.get("image"))
        if img_url:
            body["imageUrl"] = img_url
        wm = (kwargs.get("waterMark") or "").strip()
        if wm:
            body["waterMark"] = wm
        return body


class RunwayExtend(_BaseRunwayNode):
    """Runway Extend — extends a previously-generated Runway 720p video by task_id."""

    RUNWAY_PATH = "/api/v1/runway/extend"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "taskId": ("STRING", {"default": "", "tooltip": "Prior Runway task_id (720p only)."}),
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


class RunwayAleph(_BaseRunwayNode):
    """Runway Aleph — text-guided V2V transformation. Input video as IMAGE batch."""

    RUNWAY_PATH = "/api/v1/aleph/generate"
    TIMEOUT_SECONDS = 1500.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A majestic eagle soaring through mountain clouds at sunset.",
                }),
                "video": ("IMAGE", {"tooltip": "Source video as IMAGE batch (N frames)."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
            },
            "optional": {
                "waterMark": ("STRING", {"default": ""}),
                "uploadCn": ("BOOLEAN", {"default": False}),
            },
        }

    def build_runway_body(self, **kwargs: Any) -> dict[str, Any]:
        video_tensor = kwargs.get("video")
        if video_tensor is None:
            raise ValueError("Runway Aleph requires video input (IMAGE batch).")
        fps = int(kwargs.get("fps", 24))
        video_url = upload_video_frames(video_tensor, fps=fps)

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "videoUrl": video_url,
            "uploadCn": bool(kwargs.get("uploadCn", False)),
        }
        wm = (kwargs.get("waterMark") or "").strip()
        if wm:
            body["waterMark"] = wm
        return body


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieRunwayGenerate": RunwayGenerate,
    "GenesisKieRunwayExtend": RunwayExtend,
    "GenesisKieRunwayAleph": RunwayAleph,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieRunwayGenerate": "Runway Gen-4 Turbo",
    "GenesisKieRunwayExtend": "Runway Extend",
    "GenesisKieRunwayAleph": "Runway Aleph (V2V)",
}
