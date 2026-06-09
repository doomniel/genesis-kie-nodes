"""Alibaba Wan video generation nodes (complete family).

16 nodes across Wan 2.7, 2.6, 2.6 Flash, 2.5, 2.2 A14B Turbo, and Wan Animate.

Note: nodes that accept reference_video (multi-video refs) take a single
video tensor for simplicity. To pass multiple ref videos, run the node
multiple times. URL strings can also be passed via external means.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor, upload_video_frames, upload_audio


_RATIOS_WAN_3 = ["16:9", "9:16", "1:1"]
_RATIOS_WAN_2 = ["16:9", "9:16"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


def _upload_first_optional(image_tensor: Any) -> str | None:
    if image_tensor is None:
        return None
    return upload_image_tensor(image_tensor[0:1])


def _upload_batch_optional(image_tensor: Any) -> list[str]:
    if image_tensor is None:
        return []
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


# ============================================================ Wan 2.7

class Wan27T2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-7-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic city street at night."}),
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "ratio": (_RATIOS_WAN_3, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, flicker"}),
                "audio": ("AUDIO", {"tooltip": "Optional audio track to pace generation."}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "ratio": kwargs["ratio"],
            "duration": int(kwargs["duration"]),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        audio = kwargs.get("audio")
        if audio is not None:
            body["audio_url"] = upload_audio(audio)
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan27I2V(BaseKieMarketVideoNode):
    """Wan 2.7 I2V — 3 sub-modes: first-only, first+last, video-continuation."""

    MODEL = "wan/2-7-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A white cat on a windowsill in warm light."}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "First-frame image."}),
                "last_frame": ("IMAGE", {"tooltip": "Last-frame image (transition mode, requires first_frame)."}),
                "first_clip": ("IMAGE", {"tooltip": "Video to continue (mutually exclusive with frames)."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, flicker, low quality"}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        ff = kwargs.get("first_frame")
        lf = kwargs.get("last_frame")
        fc = kwargs.get("first_clip")

        if fc is not None and (ff is not None or lf is not None):
            raise ValueError("Wan 2.7 I2V: video continuation mutually exclusive with frame mode.")
        if ff is None and fc is None:
            raise ValueError("Wan 2.7 I2V: must provide first_frame or first_clip.")
        if lf is not None and ff is None:
            raise ValueError("Wan 2.7 I2V: last_frame requires first_frame.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        if ff is not None:
            body["first_frame_url"] = _upload_first(ff)
        if lf is not None:
            body["last_frame_url"] = _upload_first(lf)
        if fc is not None:
            fps = int(kwargs.get("fps", 24))
            body["first_clip_url"] = upload_video_frames(fc, fps=fps)
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan27VideoEdit(BaseKieMarketVideoNode):
    """Wan 2.7 video edit via natural-language instruction."""

    MODEL = "wan/2-7-videoedit"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Change the outfit."}),
                "video": ("IMAGE", {"tooltip": "Source video as IMAGE batch."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "aspect_ratio": (_RATIOS_WAN_3, {"default": "16:9"}),
            },
            "optional": {
                "reference_image": ("IMAGE", {"tooltip": "Optional reference image."}),
                "duration": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1}),
                "audio_setting": (["auto", "keep", "mute", "regenerate"], {"default": "auto"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low resolution, low quality"}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        video_tensor = kwargs.get("video")
        if video_tensor is None:
            raise ValueError("Wan 2.7 Video Edit requires video.")
        fps = int(kwargs.get("fps", 24))
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "video_url": upload_video_frames(video_tensor, fps=fps),
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs.get("duration", 0)),
            "audio_setting": kwargs.get("audio_setting", "auto"),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        ref_url = _upload_first_optional(kwargs.get("reference_image"))
        if ref_url:
            body["reference_image"] = ref_url
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan27R2V(BaseKieMarketVideoNode):
    """Wan 2.7 reference-to-video. Up to 5 refs total (images + 1 video)."""

    MODEL = "wan/2-7-r2v"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Image 1 eating, video 1 singing."}),
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "aspect_ratio": (_RATIOS_WAN_3, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s). Batch for multi-ref."}),
                "reference_video": ("IMAGE", {"tooltip": "Optional reference video (single, IMAGE batch)."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "first_frame": ("IMAGE", {"tooltip": "Optional first-frame anchor."}),
                "reference_voice": ("AUDIO", {"tooltip": "Optional voice clip."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality, errors"}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img_urls = _upload_batch_optional(kwargs.get("reference_images"))
        ref_video = kwargs.get("reference_video")
        n_refs = len(img_urls) + (1 if ref_video is not None else 0)
        if n_refs == 0:
            raise ValueError("Wan 2.7 R2V: at least one reference required.")
        if n_refs > 5:
            raise ValueError(f"Wan 2.7 R2V: combined refs > 5: {n_refs}.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        if img_urls:
            body["reference_image"] = img_urls
        if ref_video is not None:
            fps = int(kwargs.get("fps", 24))
            body["reference_video"] = [upload_video_frames(ref_video, fps=fps)]
        first_url = _upload_first_optional(kwargs.get("first_frame"))
        if first_url:
            body["first_frame"] = first_url
        voice = kwargs.get("reference_voice")
        if voice is not None:
            body["reference_voice"] = upload_audio(voice)
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ============================================================ Wan 2.6

class Wan26T2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-6-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Hyperrealistic ASMR scene."}),
                "duration": (["5", "10", "15"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
        }


class Wan26I2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-6-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (min 256x256)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle animation."}),
                "duration": (["5", "10", "15"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_urls": [_upload_first(kwargs.get("image"))],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class Wan26V2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-6-video-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Source video as IMAGE batch."}),
                "prompt": ("STRING", {"multiline": True, "default": "Apply cinematic grading."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = kwargs.get("video")
        if v is None:
            raise ValueError("Wan 2.6 V2V requires video.")
        fps = int(kwargs.get("fps", 24))
        return {
            "prompt": kwargs["prompt"],
            "video_urls": [upload_video_frames(v, fps=fps)],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
        }


# ============================================================ Wan 2.6 Flash

class Wan26FlashI2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-6-flash-image-to-video"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image (min 256x256)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle animation."}),
                "duration": (["5", "10", "15"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
                "audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "multi_shots": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_urls": [_upload_first(kwargs.get("image"))],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "audio": bool(kwargs.get("audio", False)),
            "multi_shots": bool(kwargs.get("multi_shots", False)),
        }


class Wan26FlashV2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-6-flash-video-to-video"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Source video as IMAGE batch."}),
                "prompt": ("STRING", {"multiline": True, "default": "Apply different style."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
            "optional": {
                "audio": ("BOOLEAN", {"default": False}),
                "multi_shots": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = kwargs.get("video")
        if v is None:
            raise ValueError("Wan 2.6 Flash V2V requires video.")
        fps = int(kwargs.get("fps", 24))
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "video_urls": [upload_video_frames(v, fps=fps)],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "multi_shots": bool(kwargs.get("multi_shots", False)),
        }
        if "audio" in kwargs:
            body["audio"] = bool(kwargs["audio"])
        return body


# ============================================================ Wan 2.5

class Wan25T2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-5-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A dimly lit jazz bar at night."}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (_RATIOS_WAN_3, {"default": "16:9"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, flicker, low quality"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "enable_prompt_expansion": bool(kwargs.get("enable_prompt_expansion", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan25I2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-5-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "First-frame image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Character speaks and smiles."}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, flicker, distorted"}),
                "enable_prompt_expansion": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "enable_prompt_expansion": bool(kwargs.get("enable_prompt_expansion", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ============================================================ Wan 2.2 A14B Turbo

class Wan22A14BT2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-2-a14b-text-to-video-turbo"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Drone shot of polar landscape."}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
                "aspect_ratio": (_RATIOS_WAN_2, {"default": "16:9"}),
            },
            "optional": {
                "enable_prompt_expansion": ("BOOLEAN", {"default": False}),
                "acceleration": (["none", "regular"], {"default": "none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "enable_prompt_expansion": bool(kwargs.get("enable_prompt_expansion", False)),
            "acceleration": kwargs.get("acceleration", "none"),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan22A14BI2V(BaseKieMarketVideoNode):
    MODEL = "wan/2-2-a14b-image-to-video-turbo"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic low-angle push-in."}),
                "resolution": (["480p", "720p"], {"default": "720p"}),
            },
            "optional": {
                "enable_prompt_expansion": ("BOOLEAN", {"default": False}),
                "acceleration": (["none", "regular"], {"default": "none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "image_url": _upload_first(kwargs.get("image")),
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "enable_prompt_expansion": bool(kwargs.get("enable_prompt_expansion", False)),
            "acceleration": kwargs.get("acceleration", "none"),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan22A14BS2V(BaseKieMarketVideoNode):
    """Wan 2.2 A14B Turbo speech-to-video: portrait + audio → lip-synced video."""

    MODEL = "wan/2-2-a14b-speech-to-video-turbo"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Portrait image."}),
                "audio": ("AUDIO", {"tooltip": "Audio file (will sync mouth)."}),
                "prompt": ("STRING", {"multiline": True, "default": "The person is talking"}),
                "resolution": (["480p", "580p", "720p"], {"default": "480p"}),
            },
            "optional": {
                "num_frames": ("INT", {"default": 80, "min": 40, "max": 120, "step": 4}),
                "frames_per_second": ("INT", {"default": 16, "min": 4, "max": 60, "step": 1}),
                "num_inference_steps": ("INT", {"default": 27, "min": 2, "max": 40, "step": 1}),
                "guidance_scale": ("FLOAT", {"default": 3.5, "min": 1.0, "max": 10.0, "step": 0.1}),
                "shift": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = kwargs.get("image")
        aud = kwargs.get("audio")
        if img is None:
            raise ValueError("Wan 2.2 S2V requires image.")
        if aud is None:
            raise ValueError("Wan 2.2 S2V requires audio.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(img),
            "audio_url": upload_audio(aud),
            "num_frames": int(kwargs.get("num_frames", 80)),
            "frames_per_second": int(kwargs.get("frames_per_second", 16)),
            "resolution": kwargs["resolution"],
            "num_inference_steps": int(kwargs.get("num_inference_steps", 27)),
            "guidance_scale": float(kwargs.get("guidance_scale", 3.5)),
            "shift": float(kwargs.get("shift", 5.0)),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ============================================================ Wan Animate

class _WanAnimateBase(BaseKieMarketVideoNode):
    """Wan Animate: motion video + character image → animated character."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "motion_video": ("IMAGE", {"tooltip": "Motion reference video (IMAGE batch)."}),
                "character_image": ("IMAGE", {"tooltip": "Character reference image."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "resolution": (["480p", "580p", "720p"], {"default": "480p"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = kwargs.get("motion_video")
        i = kwargs.get("character_image")
        if v is None:
            raise ValueError(f"{type(self).__name__} requires motion_video.")
        if i is None:
            raise ValueError(f"{type(self).__name__} requires character_image.")
        fps = int(kwargs.get("fps", 24))
        return {
            "video_url": upload_video_frames(v, fps=fps),
            "image_url": _upload_first(i),
            "resolution": kwargs["resolution"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class WanAnimateMove(_WanAnimateBase):
    MODEL = "wan/2-2-animate-move"


class WanAnimateReplace(_WanAnimateBase):
    MODEL = "wan/2-2-animate-replace"


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieWan27T2V": Wan27T2V,
    "GenesisKieWan27I2V": Wan27I2V,
    "GenesisKieWan27VideoEdit": Wan27VideoEdit,
    "GenesisKieWan27R2V": Wan27R2V,
    "GenesisKieWan26T2V": Wan26T2V,
    "GenesisKieWan26I2V": Wan26I2V,
    "GenesisKieWan26V2V": Wan26V2V,
    "GenesisKieWan26FlashI2V": Wan26FlashI2V,
    "GenesisKieWan26FlashV2V": Wan26FlashV2V,
    "GenesisKieWan25T2V": Wan25T2V,
    "GenesisKieWan25I2V": Wan25I2V,
    "GenesisKieWan22A14BT2V": Wan22A14BT2V,
    "GenesisKieWan22A14BI2V": Wan22A14BI2V,
    "GenesisKieWan22A14BS2V": Wan22A14BS2V,
    "GenesisKieWanAnimateMove": WanAnimateMove,
    "GenesisKieWanAnimateReplace": WanAnimateReplace,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieWan27T2V": "Wan 2.7 (T2V)",
    "GenesisKieWan27I2V": "Wan 2.7 (I2V)",
    "GenesisKieWan27VideoEdit": "Wan 2.7 Video Edit",
    "GenesisKieWan27R2V": "Wan 2.7 Reference-to-Video",
    "GenesisKieWan26T2V": "Wan 2.6 (T2V)",
    "GenesisKieWan26I2V": "Wan 2.6 (I2V)",
    "GenesisKieWan26V2V": "Wan 2.6 (V2V)",
    "GenesisKieWan26FlashI2V": "Wan 2.6 Flash (I2V)",
    "GenesisKieWan26FlashV2V": "Wan 2.6 Flash (V2V)",
    "GenesisKieWan25T2V": "Wan 2.5 (T2V)",
    "GenesisKieWan25I2V": "Wan 2.5 (I2V)",
    "GenesisKieWan22A14BT2V": "Wan 2.2 A14B Turbo (T2V)",
    "GenesisKieWan22A14BI2V": "Wan 2.2 A14B Turbo (I2V)",
    "GenesisKieWan22A14BS2V": "Wan 2.2 A14B Turbo Speech",
    "GenesisKieWanAnimateMove": "Wan Animate Move",
    "GenesisKieWanAnimateReplace": "Wan Animate Replace",
}
