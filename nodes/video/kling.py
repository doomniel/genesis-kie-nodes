"""Kling video generation nodes (complete family).

14 nodes across Kling 3.0, 2.6, 2.5 Turbo Pro, 2.1 family, Motion Control, and Avatar.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode
from ...client.upload import upload_image_tensor, upload_video_frames, upload_audio


_ASPECT_RATIOS_3 = ["16:9", "9:16", "1:1"]


def _upload_first(image_tensor: Any) -> str:
    if image_tensor is None or not hasattr(image_tensor, "shape"):
        raise ValueError("image tensor required")
    return upload_image_tensor(image_tensor[0:1])


def _upload_first_optional(image_tensor: Any) -> str | None:
    if image_tensor is None:
        return None
    return upload_image_tensor(image_tensor[0:1])


# ============================================================ Kling 3.0

class _Kling30Base(BaseKieMarketVideoNode):
    """Shared scaffolding for Kling 3.0 single-shot generation."""

    MODEL: ClassVar[str] = "kling-3.0/video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "In a bright rehearsal room, sunlight streams through the window.",
                }),
                "mode": (["std", "pro", "4K"], {"default": "pro"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
                "duration": ([str(n) for n in range(3, 16)], {"default": "5"}),
                "sound": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Optional first-frame image."}),
                "end_image": ("IMAGE", {"tooltip": "Optional last-frame image."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "mode": kwargs["mode"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": str(kwargs["duration"]),
            "sound": bool(kwargs.get("sound", False)),
            "multi_shots": False,
        }
        first_url = _upload_first_optional(kwargs.get("start_image"))
        last_url = _upload_first_optional(kwargs.get("end_image"))
        image_urls: list[str] = []
        if first_url:
            image_urls.append(first_url)
        if last_url:
            image_urls.append(last_url)
        if image_urls:
            body["image_urls"] = image_urls
        return body


class Kling30(_Kling30Base):
    pass


class Kling30MultiShot(BaseKieMarketVideoNode):
    """Kling 3.0 multi-shot — script with one shot per line as '<duration>:<prompt>'."""

    MODEL: ClassVar[str] = "kling-3.0/video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "shots_script": ("STRING", {
                    "multiline": True,
                    "default": "3:A happy dog running in the park\n3:The dog jumps over a fence",
                    "tooltip": "One shot per line: <duration>:<prompt>. Max 5 shots, total 3-15s.",
                }),
                "mode": (["std", "pro", "4K"], {"default": "pro"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
            },
            "optional": {
                "start_image": ("IMAGE", {"tooltip": "Optional first-frame image."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        script: str = kwargs["shots_script"]

        multi_prompt: list[dict[str, Any]] = []
        for line in script.splitlines():
            line = line.strip()
            if not line or ":" not in line:
                continue
            dur_str, _, prompt_text = line.partition(":")
            try:
                duration = int(dur_str.strip())
            except ValueError:
                continue
            multi_prompt.append({"prompt": prompt_text.strip(), "duration": duration})

        if not multi_prompt:
            raise ValueError("Kling 3.0 multi-shot needs at least one valid line.")
        if len(multi_prompt) > 5:
            raise ValueError(f"Kling 3.0 supports max 5 shots; got {len(multi_prompt)}.")
        total_duration = sum(s["duration"] for s in multi_prompt)
        if not 3 <= total_duration <= 15:
            raise ValueError(f"Total shot duration must be 3-15s; got {total_duration}s.")

        body: dict[str, Any] = {
            "prompt": "",
            "mode": kwargs["mode"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": str(total_duration),
            "sound": True,
            "multi_shots": True,
            "multi_prompt": multi_prompt,
        }
        first_url = _upload_first_optional(kwargs.get("start_image"))
        if first_url:
            body["image_urls"] = [first_url]
        return body


# ============================================================ Motion control

class _KlingMotionControlBase(BaseKieMarketVideoNode):
    """Motion control: transfer motion from a reference video onto a character image."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "input_image": ("IMAGE", {
                    "tooltip": "Subject image (head/shoulders/torso). Min 340px.",
                }),
                "motion_video": ("IMAGE", {
                    "tooltip": "Motion reference video as IMAGE batch (3-30s).",
                }),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "mode": (["std", "pro"], {"default": "pro"}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
                "character_orientation": (["video", "image"], {"default": "video"}),
                "background_source": (["input_video", "input_image"], {"default": "input_video"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        input_url = _upload_first(kwargs.get("input_image"))
        video_tensor = kwargs.get("motion_video")
        if video_tensor is None:
            raise ValueError("Kling motion-control requires motion_video.")
        fps = int(kwargs.get("fps", 24))
        video_url = upload_video_frames(video_tensor, fps=fps)

        body: dict[str, Any] = {
            "input_urls": [input_url],
            "video_urls": [video_url],
            "mode": kwargs.get("mode", "pro"),
        }
        prompt = (kwargs.get("prompt") or "").strip()
        if prompt:
            body["prompt"] = prompt
        body["character_orientation"] = kwargs.get("character_orientation", "video")
        body["background_source"] = kwargs.get("background_source", "input_video")
        return body


class Kling30MotionControl(_KlingMotionControlBase):
    MODEL = "kling-3.0/motion-control"


class Kling26MotionControl(_KlingMotionControlBase):
    MODEL = "kling-2.6/motion-control"


# ============================================================ Kling 2.6

class Kling26T2V(BaseKieMarketVideoNode):
    MODEL = "kling-2.6/text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic establishing shot at golden hour."}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
                "sound": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
            "aspect_ratio": kwargs["aspect_ratio"],
            "sound": bool(kwargs.get("sound", False)),
        }


class Kling26I2V(BaseKieMarketVideoNode):
    MODEL = "kling-2.6/image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle animation, gentle motion."}),
                "duration": (["5", "10"], {"default": "5"}),
                "sound": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        return {
            "prompt": kwargs["prompt"],
            "image_urls": [_upload_first(kwargs.get("image"))],
            "duration": str(kwargs["duration"]),
            "sound": bool(kwargs.get("sound", False)),
        }


# ============================================================ Kling 2.5 Turbo Pro

class _Kling25TurboBase(BaseKieMarketVideoNode):
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic establishing shot at sunset."}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blur, distort, and low quality"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
            "aspect_ratio": kwargs["aspect_ratio"],
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        cfg = kwargs.get("cfg_scale")
        if cfg is not None:
            body["cfg_scale"] = float(cfg)
        return body


class Kling25TurboProT2V(_Kling25TurboBase):
    MODEL = "kling/v2-5-turbo-text-to-video-pro"


class Kling25TurboProI2V(_Kling25TurboBase):
    MODEL = "kling/v2-5-turbo-image-to-video-pro"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"] = {
            "image": ("IMAGE", {"tooltip": "First-frame image."}),
            **schema["required"],
        }
        schema["optional"]["end_image"] = ("IMAGE", {"tooltip": "Optional last-frame image."})
        return schema

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        body["image_url"] = _upload_first(kwargs.get("image"))
        end_url = _upload_first_optional(kwargs.get("end_image"))
        if end_url:
            body["end_image_url"] = end_url
        return body


# ============================================================ Kling 2.1

class _Kling21ImageBase(BaseKieMarketVideoNode):
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "First-frame image."}),
                "prompt": ("STRING", {"multiline": True, "default": "Subtle cinematic animation."}),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blur, distort, and low quality"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": _upload_first(kwargs.get("image")),
            "duration": str(kwargs["duration"]),
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        cfg = kwargs.get("cfg_scale")
        if cfg is not None:
            body["cfg_scale"] = float(cfg)
        return body


class Kling21Standard(_Kling21ImageBase):
    MODEL = "kling/v2-1-standard"


class Kling21Pro(_Kling21ImageBase):
    MODEL = "kling/v2-1-pro"


class Kling21MasterI2V(_Kling21ImageBase):
    MODEL = "kling/v2-1-master-image-to-video"


class Kling21MasterT2V(BaseKieMarketVideoNode):
    MODEL = "kling/v2-1-master-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic establishing shot at sunset."}),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": "blur, distort, and low quality"}),
                "cfg_scale": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
            "aspect_ratio": kwargs["aspect_ratio"],
        }
        neg = (kwargs.get("negative_prompt") or "").strip()
        if neg:
            body["negative_prompt"] = neg
        cfg = kwargs.get("cfg_scale")
        if cfg is not None:
            body["cfg_scale"] = float(cfg)
        return body


# ============================================================ Kling AI Avatar

class _KlingAvatarBase(BaseKieMarketVideoNode):
    """Avatar lip-sync: portrait image + audio."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Avatar portrait image."}),
                "audio": ("AUDIO", {"tooltip": "Voice audio (will sync mouth to it)."}),
            },
            "optional": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        img = kwargs.get("image")
        aud = kwargs.get("audio")
        if img is None:
            raise ValueError(f"{type(self).__name__} requires image.")
        if aud is None:
            raise ValueError(f"{type(self).__name__} requires audio.")
        return {
            "image_url": _upload_first(img),
            "audio_url": upload_audio(aud),
            "prompt": kwargs.get("prompt", "") or "",
        }


class KlingAvatarStandard(_KlingAvatarBase):
    MODEL = "kling/ai-avatar-standard"


class KlingAvatarPro(_KlingAvatarBase):
    MODEL = "kling/ai-avatar-pro"


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieKling30": Kling30,
    "GenesisKieKling30MultiShot": Kling30MultiShot,
    "GenesisKieKling30MotionControl": Kling30MotionControl,
    "GenesisKieKling26T2V": Kling26T2V,
    "GenesisKieKling26I2V": Kling26I2V,
    "GenesisKieKling26MotionControl": Kling26MotionControl,
    "GenesisKieKling25TurboProT2V": Kling25TurboProT2V,
    "GenesisKieKling25TurboProI2V": Kling25TurboProI2V,
    "GenesisKieKling21Standard": Kling21Standard,
    "GenesisKieKling21Pro": Kling21Pro,
    "GenesisKieKling21MasterT2V": Kling21MasterT2V,
    "GenesisKieKling21MasterI2V": Kling21MasterI2V,
    "GenesisKieKlingAvatarStandard": KlingAvatarStandard,
    "GenesisKieKlingAvatarPro": KlingAvatarPro,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieKling30": "Kling 3.0",
    "GenesisKieKling30MultiShot": "Kling 3.0 (multi-shot)",
    "GenesisKieKling30MotionControl": "Kling 3.0 Motion Control",
    "GenesisKieKling26T2V": "Kling 2.6 (T2V)",
    "GenesisKieKling26I2V": "Kling 2.6 (I2V)",
    "GenesisKieKling26MotionControl": "Kling 2.6 Motion Control",
    "GenesisKieKling25TurboProT2V": "Kling 2.5 Turbo Pro (T2V)",
    "GenesisKieKling25TurboProI2V": "Kling 2.5 Turbo Pro (I2V)",
    "GenesisKieKling21Standard": "Kling 2.1 Standard",
    "GenesisKieKling21Pro": "Kling 2.1 Pro",
    "GenesisKieKling21MasterT2V": "Kling 2.1 Master (T2V)",
    "GenesisKieKling21MasterI2V": "Kling 2.1 Master (I2V)",
    "GenesisKieKlingAvatarStandard": "Kling AI Avatar Standard",
    "GenesisKieKlingAvatarPro": "Kling AI Avatar Pro",
}
