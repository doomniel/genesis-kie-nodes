"""HappyHorse 1.0 video generation nodes (via GenesisLab proxy / Kie.ai)."""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor, upload_video_frames


_RATIOS = ["16:9", "9:16", "1:1", "4:3", "3:4"]
_RESOLUTIONS = ["720P", "1080P"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


def _upload_batch(image_tensor: Any) -> list[str]:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


def _upload_batch_optional(image_tensor: Any) -> list[str]:
    if image_tensor is None:
        return []
    return _upload_batch(image_tensor)


class HappyHorseT2V(BaseKieMarketVideoNode):
    MODEL = "happyhorse/text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A silver horse runs through a rain-soaked neon alley.",
                }),
                "resolution": (_RESOLUTIONS, {"default": "1080P"}),
                "aspect_ratio": (_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
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


class HappyHorseI2V(BaseKieMarketVideoNode):
    MODEL = "happyhorse/image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "First-frame image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic motion."}),
                "resolution": (_RESOLUTIONS, {"default": "1080P"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class HappyHorseR2V(BaseKieMarketVideoNode):
    MODEL = "happyhorse/reference-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Reference image(s). Batch for multi-ref (1-9). Use 'character1','character2' in prompt.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "character1 walks toward the camera in a cinematic outdoor scene.",
                }),
                "resolution": (_RESOLUTIONS, {"default": "1080P"}),
                "aspect_ratio": (_RATIOS, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        urls = _upload_batch(kwargs.get("images"))
        if not urls:
            raise ValueError("HappyHorse R2V requires at least one reference image.")
        if len(urls) > 9:
            raise ValueError(f"HappyHorse R2V: max 9 references, got {len(urls)}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_urls": urls,
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class HappyHorseEdit(BaseKieMarketVideoNode):
    """HappyHorse 1.0 V2V edit. Source video (IMAGE batch) + optional reference images."""

    MODEL = "happyhorse/video-edit"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Source video as IMAGE batch (N frames)."}),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Apply cinematic color grading and a slow camera push-in.",
                }),
                "resolution": (_RESOLUTIONS, {"default": "1080P"}),
                "audio_setting": (["auto", "origin"], {"default": "auto"}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
            },
            "optional": {
                "reference_images": ("IMAGE", {
                    "tooltip": "Optional reference image(s). Batch for multi-ref (0-5).",
                }),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        video_tensor = kwargs.get("video")
        if video_tensor is None:
            raise ValueError("HappyHorse Edit requires video input (IMAGE batch).")
        fps = int(kwargs.get("fps", 24))
        video_url = upload_video_frames(video_tensor, fps=fps)

        refs = _upload_batch_optional(kwargs.get("reference_images"))
        if len(refs) > 5:
            raise ValueError(f"HappyHorse Edit: max 5 references, got {len(refs)}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "video_url": video_url,
            "resolution": kwargs["resolution"],
            "audio_setting": kwargs.get("audio_setting", "auto"),
        }
        if refs:
            body["image_urls"] = refs
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieHappyHorseT2V": HappyHorseT2V,
    "GenesisKieHappyHorseI2V": HappyHorseI2V,
    "GenesisKieHappyHorseR2V": HappyHorseR2V,
    "GenesisKieHappyHorseEdit": HappyHorseEdit,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieHappyHorseT2V": "HappyHorse 1.0 (T2V)",
    "GenesisKieHappyHorseI2V": "HappyHorse 1.0 (I2V)",
    "GenesisKieHappyHorseR2V": "HappyHorse 1.0 Reference-to-Video",
    "GenesisKieHappyHorseEdit": "HappyHorse 1.0 Video Edit",
}
