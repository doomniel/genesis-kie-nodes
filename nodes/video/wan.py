"""Alibaba Wan video generation nodes (complete family).

Covers all 16 Wan endpoints in Kie.ai:

Wan 2.7 (4): T2V, I2V, Video Edit, Reference-to-Video
Wan 2.6 (3): T2V, I2V, V2V
Wan 2.6 Flash (2): I2V, V2V
Wan 2.5 (2): T2V, I2V
Wan 2.2 A14B Turbo (3): T2V, I2V, Speech-to-Video
Wan Animate (2): Move, Replace

All use the Market endpoint /api/v1/jobs/createTask.

Parameter schemas extracted verbatim from docs.kie.ai OpenAPI specs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


_RATIOS_WAN_3 = ["16:9", "9:16", "1:1"]
_RATIOS_WAN_2 = ["16:9", "9:16"]


# ============================================================ Wan 2.7

class Wan27T2V(BaseKieMarketVideoNode):
    """Wan 2.7 text-to-video. Supports optional audio_url to pace generation."""

    MODEL = "wan/2-7-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A futuristic city street at night.",
                }),
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "ratio": (_RATIOS_WAN_3, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, low quality, flicker"}),
                "audio_url": ("STRING", {"default": "", "tooltip": "Optional audio track URL."}),
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
        audio = (kwargs.get("audio_url") or "").strip()
        if audio:
            body["audio_url"] = audio
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan27I2V(BaseKieMarketVideoNode):
    """Wan 2.7 image-to-video. 3 sub-modes: first-only, first+last, video-continuation."""

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
            },
            "optional": {
                "first_frame_url": ("STRING", {"default": "", "tooltip": "First-frame image URL."}),
                "last_frame_url": ("STRING", {"default": "", "tooltip": "Last-frame URL (transition mode)."}),
                "first_clip_url": ("STRING", {"default": "", "tooltip": "Video to continue (mutually exclusive with frames)."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "blurry, flicker, low quality"}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        ff = (kwargs.get("first_frame_url") or "").strip()
        lf = (kwargs.get("last_frame_url") or "").strip()
        fc = (kwargs.get("first_clip_url") or "").strip()

        if fc and (ff or lf):
            raise ValueError("Wan 2.7 I2V: video continuation mutually exclusive with frame mode.")
        if not (ff or fc):
            raise ValueError("Wan 2.7 I2V: must provide first_frame_url or first_clip_url.")
        if lf and not ff:
            raise ValueError("Wan 2.7 I2V: last_frame_url requires first_frame_url.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        if ff:
            body["first_frame_url"] = ff
        if lf:
            body["last_frame_url"] = lf
        if fc:
            body["first_clip_url"] = fc
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
                "video_url": ("STRING", {"default": "", "tooltip": "Source video URL (required)."}),
                "resolution": (["480p", "720p", "1080p"], {"default": "1080p"}),
                "aspect_ratio": (_RATIOS_WAN_3, {"default": "16:9"}),
            },
            "optional": {
                "reference_image": ("STRING", {"default": "", "tooltip": "Optional reference image URL."}),
                "duration": ("INT", {"default": 0, "min": 0, "max": 15, "step": 1, "tooltip": "0 = match input."}),
                "audio_setting": (["auto", "keep", "mute", "regenerate"], {"default": "auto"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low resolution, low quality"}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        if not v:
            raise ValueError("Wan 2.7 Video Edit requires video_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "video_url": v,
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs.get("duration", 0)),
            "audio_setting": kwargs.get("audio_setting", "auto"),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        ref = (kwargs.get("reference_image") or "").strip()
        if ref:
            body["reference_image"] = ref
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class Wan27R2V(BaseKieMarketVideoNode):
    """Wan 2.7 reference-to-video. Up to 5 refs (images+videos combined)."""

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
                "reference_image_urls": ("STRING", {"default": "", "tooltip": "Comma-separated image URLs."}),
                "reference_video_urls": ("STRING", {"default": "", "tooltip": "Comma-separated video URLs."}),
                "first_frame_url": ("STRING", {"default": "", "tooltip": "Optional first-frame anchor."}),
                "reference_voice_url": ("STRING", {"default": "", "tooltip": "Optional voice clip URL."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "low quality, errors"}),
                "prompt_extend": ("BOOLEAN", {"default": True}),
                "watermark": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        imgs = self._csv((kwargs.get("reference_image_urls") or "").strip())
        vids = self._csv((kwargs.get("reference_video_urls") or "").strip())

        if len(imgs) + len(vids) > 5:
            raise ValueError(f"Wan 2.7 R2V: combined refs > 5: {len(imgs) + len(vids)}.")
        if not (imgs or vids):
            raise ValueError("Wan 2.7 R2V: at least one reference required.")

        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
            "prompt_extend": bool(kwargs.get("prompt_extend", True)),
            "watermark": bool(kwargs.get("watermark", False)),
        }
        if imgs:
            body["reference_image"] = imgs
        if vids:
            body["reference_video"] = vids
        first = (kwargs.get("first_frame_url") or "").strip()
        if first:
            body["first_frame"] = first
        voice = (kwargs.get("reference_voice_url") or "").strip()
        if voice:
            body["reference_voice"] = voice
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Wan 2.6

class Wan26T2V(BaseKieMarketVideoNode):
    """Wan 2.6 text-to-video (720p/1080p; 5/10/15s)."""

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
    """Wan 2.6 image-to-video. Min image 256x256."""

    MODEL = "wan/2-6-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Input image URL (min 256x256)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle animation."}),
                "duration": (["5", "10", "15"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Wan 2.6 I2V requires image_url.")
        return {
            "prompt": kwargs["prompt"],
            "image_urls": [img],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class Wan26V2V(BaseKieMarketVideoNode):
    """Wan 2.6 video-to-video. Transform an existing video with a new prompt."""

    MODEL = "wan/2-6-video-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "tooltip": "Input video URL (MP4/MOV/MKV, max 10MB)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Apply cinematic grading."}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        if not v:
            raise ValueError("Wan 2.6 V2V requires video_url.")
        return {
            "prompt": kwargs["prompt"],
            "video_urls": [v],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
        }


# ============================================================ Wan 2.6 Flash

class Wan26FlashI2V(BaseKieMarketVideoNode):
    """Wan 2.6 Flash image-to-video — cheaper/faster than 2.6."""

    MODEL = "wan/2-6-flash-image-to-video"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Input image URL (min 256x256)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle animation."}),
                "duration": (["5", "10", "15"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
                "audio": ("BOOLEAN", {"default": False, "tooltip": "With audio (affects cost)."}),
            },
            "optional": {
                "multi_shots": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Wan 2.6 Flash I2V requires image_url.")
        return {
            "prompt": kwargs["prompt"],
            "image_urls": [img],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "audio": bool(kwargs.get("audio", False)),
            "multi_shots": bool(kwargs.get("multi_shots", False)),
        }


class Wan26FlashV2V(BaseKieMarketVideoNode):
    """Wan 2.6 Flash video-to-video — cheaper/faster than 2.6 V2V."""

    MODEL = "wan/2-6-flash-video-to-video"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "tooltip": "Input video URL."}),
                "prompt": ("STRING", {"multiline": True, "default": "Apply different style."}),
                "duration": (["5", "10"], {"default": "5"}),
                "resolution": (["720p", "1080p"], {"default": "1080p"}),
            },
            "optional": {
                "audio": ("BOOLEAN", {"default": False}),
                "multi_shots": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        if not v:
            raise ValueError("Wan 2.6 Flash V2V requires video_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "video_urls": [v],
            "duration": str(kwargs["duration"]),
            "resolution": kwargs["resolution"],
            "multi_shots": bool(kwargs.get("multi_shots", False)),
        }
        if "audio" in kwargs:
            body["audio"] = bool(kwargs["audio"])
        return body


# ============================================================ Wan 2.5

class Wan25T2V(BaseKieMarketVideoNode):
    """Wan 2.5 text-to-video. Max prompt 800 chars."""

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
    """Wan 2.5 image-to-video (uses image_url field, not array)."""

    MODEL = "wan/2-5-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "First-frame image URL (required)."}),
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
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Wan 2.5 I2V requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
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
    """Wan 2.2 A14B Turbo text-to-video (open-source 14B params)."""

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
                "acceleration": (["none", "regular"], {"default": "none", "tooltip": "Higher = faster, lower quality."}),
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
    """Wan 2.2 A14B Turbo image-to-video."""

    MODEL = "wan/2-2-a14b-image-to-video-turbo"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Input image URL (required)."}),
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
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Wan 2.2 A14B I2V requires image_url.")
        body: dict[str, Any] = {
            "image_url": img,
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
    """Wan 2.2 A14B Turbo speech-to-video (lip-synced from audio).

    Exposes fine-grained controls: num_frames (40-120 step 4), fps (4-60),
    inference_steps (2-40), guidance_scale, shift.
    """

    MODEL = "wan/2-2-a14b-speech-to-video-turbo"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Portrait image URL (required)."}),
                "audio_url": ("STRING", {"default": "", "tooltip": "Audio file URL (max 10MB)."}),
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
        img = (kwargs.get("image_url") or "").strip()
        aud = (kwargs.get("audio_url") or "").strip()
        if not img:
            raise ValueError("Wan 2.2 Speech-to-Video requires image_url.")
        if not aud:
            raise ValueError("Wan 2.2 Speech-to-Video requires audio_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "audio_url": aud,
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
    """Shared scaffolding for Wan Animate Move/Replace.

    Both take motion video + character image and apply motion to character.
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1200.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "video_url": ("STRING", {"default": "", "tooltip": "Motion reference video (MP4/MOV/MKV)."}),
                "image_url": ("STRING", {"default": "", "tooltip": "Character reference image."}),
                "resolution": (["480p", "580p", "720p"], {"default": "480p"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        v = (kwargs.get("video_url") or "").strip()
        i = (kwargs.get("image_url") or "").strip()
        if not v:
            raise ValueError(f"{type(self).__name__} requires video_url.")
        if not i:
            raise ValueError(f"{type(self).__name__} requires image_url.")
        return {
            "video_url": v,
            "image_url": i,
            "resolution": kwargs["resolution"],
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


class WanAnimateMove(_WanAnimateBase):
    """Wan Animate Move — make character in image perform motion from video."""
    MODEL = "wan/2-2-animate-move"


class WanAnimateReplace(_WanAnimateBase):
    """Wan Animate Replace — swap the character in a video with the image."""
    MODEL = "wan/2-2-animate-replace"


# ----------------------------------------------------------------- Registration

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
    "GenesisKieWan27T2V": "Kie — Wan 2.7 (T2V)",
    "GenesisKieWan27I2V": "Kie — Wan 2.7 (I2V)",
    "GenesisKieWan27VideoEdit": "Kie — Wan 2.7 Video Edit",
    "GenesisKieWan27R2V": "Kie — Wan 2.7 Reference-to-Video",
    "GenesisKieWan26T2V": "Kie — Wan 2.6 (T2V)",
    "GenesisKieWan26I2V": "Kie — Wan 2.6 (I2V)",
    "GenesisKieWan26V2V": "Kie — Wan 2.6 (V2V)",
    "GenesisKieWan26FlashI2V": "Kie — Wan 2.6 Flash (I2V)",
    "GenesisKieWan26FlashV2V": "Kie — Wan 2.6 Flash (V2V)",
    "GenesisKieWan25T2V": "Kie — Wan 2.5 (T2V)",
    "GenesisKieWan25I2V": "Kie — Wan 2.5 (I2V)",
    "GenesisKieWan22A14BT2V": "Kie — Wan 2.2 A14B Turbo (T2V)",
    "GenesisKieWan22A14BI2V": "Kie — Wan 2.2 A14B Turbo (I2V)",
    "GenesisKieWan22A14BS2V": "Kie — Wan 2.2 A14B Turbo Speech",
    "GenesisKieWanAnimateMove": "Kie — Wan Animate Move",
    "GenesisKieWanAnimateReplace": "Kie — Wan Animate Replace",
}
