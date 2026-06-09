"""ByteDance video generation nodes (complete family). 8 nodes."""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor, upload_video_frames, upload_audio


_RATIOS_SEEDANCE = ["16:9", "9:16", "1:1"]
_RATIOS_V1 = ["21:9", "16:9", "4:3", "1:1", "3:4", "9:16"]
_RATIOS_V1_LITE = ["16:9", "4:3", "1:1", "3:4", "9:16", "9:21"]


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


# ============================================================ Seedance 2.x

class _Seedance2Base(BaseKieMarketVideoNode):
    """Seedance 2.0 base. Multiple modes via optional inputs.

    Note: reference_video accepts a single video (IMAGE batch). For multiple
    ref videos, use external URLs (not currently supported by this node).
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0
    SUPPORTED_RESOLUTIONS: ClassVar[list[str]] = ["480p", "720p", "1080p"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A serene beach at sunset with waves."}),
                "resolution": (cls.SUPPORTED_RESOLUTIONS, {"default": "720p"}),
                "aspect_ratio": (_RATIOS_SEEDANCE, {"default": "16:9"}),
                "duration": ("INT", {"default": 5, "min": 3, "max": 15, "step": 1}),
                "generate_audio": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "first_frame": ("IMAGE", {"tooltip": "First-frame image (I2V mode)."}),
                "last_frame": ("IMAGE", {"tooltip": "Last-frame image (transition mode, requires first_frame)."}),
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s). Batch for multi-ref."}),
                "reference_video": ("IMAGE", {"tooltip": "Single reference video as IMAGE batch."}),
                "reference_audio": ("AUDIO", {"tooltip": "Reference audio."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
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
        first = kwargs.get("first_frame")
        last = kwargs.get("last_frame")
        ref_i = kwargs.get("reference_images")
        ref_v = kwargs.get("reference_video")
        ref_a = kwargs.get("reference_audio")

        has_frame_mode = first is not None or last is not None
        has_ref_mode = ref_i is not None or ref_v is not None or ref_a is not None

        if has_frame_mode and has_ref_mode:
            raise ValueError("Seedance: frame mode and reference mode are mutually exclusive.")

        if first is not None:
            body["first_frame_url"] = _upload_first(first)
        if last is not None:
            body["last_frame_url"] = _upload_first(last)

        ref_img_urls = _upload_batch_optional(ref_i)
        if ref_img_urls:
            body["reference_image_urls"] = ref_img_urls
        if ref_v is not None:
            fps = int(kwargs.get("fps", 24))
            body["reference_video_urls"] = [upload_video_frames(ref_v, fps=fps)]
        if ref_a is not None:
            body["reference_audio_urls"] = [upload_audio(ref_a)]

        return body


class Seedance20(_Seedance2Base):
    MODEL = "bytedance/seedance-2"
    SUPPORTED_RESOLUTIONS = ["480p", "720p", "1080p"]


class Seedance20Fast(_Seedance2Base):
    MODEL = "bytedance/seedance-2-fast"
    SUPPORTED_RESOLUTIONS = ["480p", "720p"]


# ============================================================ Seedance 1.5 Pro

class Seedance15Pro(BaseKieMarketVideoNode):
    """Seedance 1.5 Pro — input_urls accepts 0-2 images."""

    MODEL = "bytedance/seedance-1.5-pro"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A serene beach at sunset."}),
                "aspect_ratio": (["1:1", "4:3", "3:4", "16:9", "9:16", "21:9"], {"default": "16:9"}),
                "resolution": (["480p", "720p", "1080p"], {"default": "720p"}),
                "duration": ([4, 8, 12], {"default": 8}),
            },
            "optional": {
                "input_images": ("IMAGE", {"tooltip": "Optional input image(s). Batch for 2 max. Empty = T2V."}),
                "fixed_lens": ("BOOLEAN", {"default": False}),
                "generate_audio": ("BOOLEAN", {"default": False}),
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
        urls = _upload_batch_optional(kwargs.get("input_images"))
        if len(urls) > 2:
            raise ValueError(f"Seedance 1.5 Pro: max 2 input images, got {len(urls)}.")
        if urls:
            body["input_urls"] = urls
        return body


# ============================================================ Bytedance V1 Pro

class BytedanceV1ProT2V(BaseKieMarketVideoNode):
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
        body["seed"] = seed if seed > 0 else -1
        return body


class BytedanceV1ProI2V(BaseKieMarketVideoNode):
    MODEL = "bytedance/v1-pro-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image."}),
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
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "camera_fixed": bool(kwargs.get("camera_fixed", False)),
            "enable_safety_checker": bool(kwargs.get("enable_safety_checker", True)),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }
        seed = int(kwargs.get("seed") or 0)
        body["seed"] = seed if seed > 0 else -1
        return body


class BytedanceV1ProFastI2V(BaseKieMarketVideoNode):
    MODEL = "bytedance/v1-pro-fast-image-to-video"
    POLL_INTERVAL_SECONDS = 4.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Animated motion."}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "nsfw_checker": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "resolution": kwargs["resolution"],
            "duration": str(kwargs["duration"]),
            "nsfw_checker": bool(kwargs.get("nsfw_checker", False)),
        }


# ============================================================ Bytedance V1 Lite

class BytedanceV1LiteT2V(BaseKieMarketVideoNode):
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
    MODEL = "bytedance/v1-lite-image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image."}),
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
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
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
    "GenesisKieSeedance20": "Seedance 2.0",
    "GenesisKieSeedance20Fast": "Seedance 2.0 Fast",
    "GenesisKieSeedance15Pro": "Seedance 1.5 Pro",
    "GenesisKieBytedanceV1ProT2V": "Bytedance V1 Pro (T2V)",
    "GenesisKieBytedanceV1ProI2V": "Bytedance V1 Pro (I2V)",
    "GenesisKieBytedanceV1ProFastI2V": "Bytedance V1 Pro Fast (I2V)",
    "GenesisKieBytedanceV1LiteT2V": "Bytedance V1 Lite (T2V)",
    "GenesisKieBytedanceV1LiteI2V": "Bytedance V1 Lite (I2V)",
}
