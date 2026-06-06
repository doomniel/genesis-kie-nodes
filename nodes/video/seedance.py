"""ByteDance video generation nodes (complete family).

Covers all 8 Bytedance endpoints in Kie.ai:

Seedance 2.x:
  - Seedance 2.0           (bytedance/seedance-2)
  - Seedance 2.0 Fast      (bytedance/seedance-2-fast)

Seedance 1.5:
  - Seedance 1.5 Pro       (bytedance/seedance-1.5-pro)

V1 Pro:
  - V1 Pro Text-to-Video        (bytedance/v1-pro-text-to-video)
  - V1 Pro Image-to-Video       (bytedance/v1-pro-image-to-video)
  - V1 Pro Fast Image-to-Video  (bytedance/v1-pro-fast-image-to-video)

V1 Lite:
  - V1 Lite Text-to-Video       (bytedance/v1-lite-text-to-video)
  - V1 Lite Image-to-Video      (bytedance/v1-lite-image-to-video)

All use the Market endpoint /api/v1/jobs/createTask.

Parameter schemas extracted verbatim from docs.kie.ai OpenAPI specs.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


_RATIOS_SEEDANCE = ["16:9", "9:16", "1:1"]
_RATIOS_V1 = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
_RATIOS_V1_LITE = ["16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]


# ============================================================ Seedance 2.x

class _Seedance2Base(BaseKieMarketVideoNode):
    """Shared scaffolding for Seedance 2.0 tiers.

    All Seedance modes (t2v, i2v, transition, multimodal reference) live
    in the same endpoint — the mode is determined by which optional
    inputs are set.
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0
    SUPPORTED_RESOLUTIONS: ClassVar[list[str]] = ["480p", "720p", "1080p"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A serene beach at sunset with waves.",
                }),
                "resolution": (cls.SUPPORTED_RESOLUTIONS, {"default": "720p"}),
                "aspect_ratio": (_RATIOS_SEEDANCE, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "first_frame_url": ("STRING", {"default": "", "tooltip": "Image-to-video mode."}),
                "last_frame_url": ("STRING", {"default": "", "tooltip": "Transition mode (requires first_frame)."}),
                "reference_image_urls": ("STRING", {"default": "", "tooltip": "Comma-separated reference images."}),
                "reference_video_urls": ("STRING", {"default": "", "tooltip": "Comma-separated reference videos."}),
                "reference_audio_urls": ("STRING", {"default": "", "tooltip": "Comma-separated reference audio."}),
                "return_last_frame": ("BOOLEAN", {"default": False}),
                "web_search": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "resolution": kwargs["resolution"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": int(kwargs["duration"]),
            "generate_audio": bool(kwargs.get("generate_audio", False)),
            "return_last_frame": bool(kwargs.get("return_last_frame", False)),
            "web_search": bool(kwargs.get("web_search", False)),
        }
        first = (kwargs.get("first_frame_url") or "").strip()
        last = (kwargs.get("last_frame_url") or "").strip()
        ref_i = self._csv((kwargs.get("reference_image_urls") or "").strip())
        ref_v = self._csv((kwargs.get("reference_video_urls") or "").strip())
        ref_a = self._csv((kwargs.get("reference_audio_urls") or "").strip())

        if (first or last) and (ref_i or ref_v or ref_a):
            raise ValueError("Seedance: first/last frame mode and reference mode are mutually exclusive.")

        if first:
            body["first_frame_url"] = first
        if last:
            body["last_frame_url"] = last
        if ref_i:
            body["reference_image_urls"] = ref_i
        if ref_v:
            body["reference_video_urls"] = ref_v
        if ref_a:
            body["reference_audio_urls"] = ref_a
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


class Seedance20(_Seedance2Base):
    """ByteDance Seedance 2.0 — full quality, up to 1080p."""
    MODEL = "bytedance/seedance-2"
    SUPPORTED_RESOLUTIONS = ["480p", "720p", "1080p"]


class Seedance20Fast(_Seedance2Base):
    """ByteDance Seedance 2.0 Fast — cheaper, up to 720p."""
    MODEL = "bytedance/seedance-2-fast"
    SUPPORTED_RESOLUTIONS = ["480p", "720p"]


# ============================================================ Seedance 1.5 Pro

class Seedance15Pro(BaseKieMarketVideoNode):
    """ByteDance Seedance 1.5 Pro.

    Different parameter shape from Seedance 2.x:
    - ``input_urls`` (array, 0-2 images, optional — text-to-video if empty)
    - ``aspect_ratio`` required (1:1 / 4:3 / 3:4 / 16:9 / 9:16 / 21:9)
    - ``duration``: 4 / 8 / 12 (integer, not free range)
    - ``fixed_lens``: lock camera for static shots
    - ``generate_audio``: optional audio (higher cost)
    """

    MODEL = "bytedance/seedance-1.5-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A serene beach at sunset.",
                }),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "21:9"], {"default": "16:9"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": ([4, 8, 12], {"default": 8}),
            },
            "optional": {
                "input_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated input image URLs (0-2). Empty = text-to-video.",
                }),
                "fixed_lens": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lock camera for stable, static shots.",
                }),
                "generate_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Generate audio (higher cost).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "duration": int(kwargs["duration"]),
            "fixed_lens": bool(kwargs.get("fixed_lens", False)),
            "generate_audio": bool(kwargs.get("generate_audio", False)),
        }
        inputs = self._csv((kwargs.get("input_urls") or "").strip())
        if len(inputs) > 2:
            raise ValueError(f"Seedance 1.5 Pro: max 2 input images, got {len(inputs)}.")
        if inputs:
            body["input_urls"] = inputs
        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Bytedance V1 Pro

class BytedanceV1ProT2V(BaseKieMarketVideoNode):
    """ByteDance V1 Pro text-to-video. Up to 21:9 aspect ratio."""

    MODEL = "bytedance/v1-pro-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic outdoor scene at sunset."}),
                "aspect_ratio": (_RATIOS_V1, {"default": "16:9"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "camera_fixed": bool(kwargs.get("camera_fixed", False)),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        else:
            body["seed"] = -1  # API default for "random"
        return body


class BytedanceV1ProI2V(BaseKieMarketVideoNode):
    """ByteDance V1 Pro image-to-video. Uses ``image_url`` (single, not array)."""

    MODEL = "bytedance/v1-pro-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Input image URL (required)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Cinematic animation."}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Bytedance V1 Pro I2V requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "camera_fixed": bool(kwargs.get("camera_fixed", False)),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        else:
            body["seed"] = -1
        return body


class BytedanceV1ProFastI2V(BaseKieMarketVideoNode):
    """ByteDance V1 Pro Fast image-to-video.

    Cheaper than V1 Pro I2V. Note: 720p/1080p only (no 480p per docs).
    """

    MODEL = "bytedance/v1-pro-fast-image-to-video"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Input image URL (required)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Animated motion."}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Bytedance V1 Pro Fast I2V requires image_url.")
        return {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


# ============================================================ Bytedance V1 Lite

class BytedanceV1LiteT2V(BaseKieMarketVideoNode):
    """ByteDance V1 Lite text-to-video. Includes 9:21 portrait ratio."""

    MODEL = "bytedance/v1-lite-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "Wide-angle shot at dawn."}),
                "aspect_ratio": (_RATIOS_V1_LITE, {"default": "16:9"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "camera_fixed": bool(kwargs.get("camera_fixed", False)),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


class BytedanceV1LiteI2V(BaseKieMarketVideoNode):
    """ByteDance V1 Lite image-to-video."""

    MODEL = "bytedance/v1-lite-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {"default": "", "tooltip": "Input image URL (required)."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle animation."}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "camera_fixed": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
                "enable_safety_checker": ("BOOLEAN", {"default": True}),
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = (kwargs.get("image_url") or "").strip()
        if not img:
            raise ValueError("Bytedance V1 Lite I2V requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": img,
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "camera_fixed": bool(kwargs.get("camera_fixed", False)),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        if seed > 0:
            body["seed"] = seed
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSeedance20": Seedance20,
    "GenesisKieSeedance20Fast": Seedance20Fast,
    "GenesisKieSeedance15Pro": Seedance15Pro,
    "GenesisKieBytedanceV1ProT2V": BytedanceV1ProT2V,
    "GenesisKieBytedanceV1ProI2V": BytedanceV1ProI2V,
    "GenesisKieBytedanceV1ProFastI2V": BytedanceV1ProFastI2V,
    "GenesisKieBytedanceV1LiteT2V": BytedanceV1LiteT2V,
    "GenesisKieBytedanceV1LiteI2V": BytedanceV1LiteI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSeedance20": "Kie — Seedance 2.0",
    "GenesisKieSeedance20Fast": "Kie — Seedance 2.0 Fast",
    "GenesisKieSeedance15Pro": "Kie — Seedance 1.5 Pro",
    "GenesisKieBytedanceV1ProT2V": "Kie — Bytedance V1 Pro (T2V)",
    "GenesisKieBytedanceV1ProI2V": "Kie — Bytedance V1 Pro (I2V)",
    "GenesisKieBytedanceV1ProFastI2V": "Kie — Bytedance V1 Pro Fast (I2V)",
    "GenesisKieBytedanceV1LiteT2V": "Kie — Bytedance V1 Lite (T2V)",
    "GenesisKieBytedanceV1LiteI2V": "Kie — Bytedance V1 Lite (I2V)",
}
