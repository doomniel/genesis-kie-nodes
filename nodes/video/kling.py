"""Kling video generation nodes (complete family).

Covers all 14 Kling endpoints in Kie.ai:

  Flagship:
    - Kling 3.0 (single-shot + multi-shot with elements)
    - Kling 3.0 Motion Control

  Workhorses:
    - Kling 2.6 (T2V + I2V)
    - Kling 2.6 Motion Control
    - Kling 2.5 Turbo Pro (T2V + I2V)

  Legacy:
    - Kling 2.1 Master (T2V + I2V)
    - Kling 2.1 Pro (image-to-video)
    - Kling 2.1 Standard (image-to-video)

  Specials:
    - Kling AI Avatar Standard (lip-sync, up to 15s, 720p)
    - Kling AI Avatar Pro (lip-sync, up to 15s, 1080p)

All use the Market endpoint /api/v1/jobs/createTask.

Pricing reference (Kie.ai, 2026):
    Kling 3.0 std 720p           $0.07/s no audio,  $0.10/s w/audio
    Kling 3.0 pro 1080p          $0.09/s no audio,  $0.135/s w/audio
    Kling 3.0 4K                 $0.335/s
    Kling 3.0 motion-control     $0.10/s 720p, $0.135/s 1080p
    Kling 2.6                    $0.275/video 5s no audio,  $0.55 w/audio
    Kling 2.6 motion-control     $0.055/s 720p, $0.09/s 1080p
    Kling 2.5 Turbo Pro 5s       $0.21/video
    Kling 2.5 Turbo Pro 10s      $0.42/video
    Kling 2.1 Standard 5s        $0.125/video
    Kling 2.1 Pro 5s             $0.25/video
    Kling 2.1 Master 5s          $0.80/video
    Kling Avatar Std 720p        $0.04/s lip-sync (up to 15s)
    Kling Avatar Pro 1080p       $0.08/s lip-sync (up to 15s)
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


# Common enums.
_ASPECT_RATIOS_3 = ["16:9", "9:16", "1:1"]


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
                    "default": "In a bright rehearsal room, sunlight streams "
                               "through the window.",
                }),
                "mode": (["std", "pro", "4K"], {"default": "pro"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
                "duration": ([str(n) for n in range(3, 16)], {"default": "5"}),
                "sound": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional first-frame image URL.",
                }),
                "last_frame_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional last-frame URL (image-to-image bridge).",
                }),
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
        first = (kwargs.get("image_url") or "").strip()
        last = (kwargs.get("last_frame_url") or "").strip()
        image_urls: list[str] = []
        if first:
            image_urls.append(first)
        if last:
            image_urls.append(last)
        if image_urls:
            body["image_urls"] = image_urls
        return body


class Kling30(_Kling30Base):
    """Kling 3.0 single-shot (text-to-video and image-to-video)."""


class Kling30MultiShot(BaseKieMarketVideoNode):
    """Kling 3.0 multi-shot — up to 5 shots, each with its own prompt + duration.

    Accepts a "scene script" string with one shot per line, formatted as
    ``<duration>:<prompt>``. Example::

        3:A happy dog running in the park
        4:The dog jumps over a fence
        3:Close-up of the dog's smiling face

    Total duration must be 3-15s.
    """

    MODEL: ClassVar[str] = "kling-3.0/video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "shots_script": ("STRING", {
                    "multiline": True,
                    "default": "3:A happy dog running in the park\n"
                               "3:The dog jumps over a fence",
                    "tooltip": "One shot per line, formatted as "
                               "<duration>:<prompt>. Max 5 shots, "
                               "each 1-12s, total 3-15s.",
                }),
                "mode": (["std", "pro", "4K"], {"default": "pro"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
            },
            "optional": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional first-frame image.",
                }),
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
            multi_prompt.append({
                "prompt": prompt_text.strip(),
                "duration": duration,
            })

        if not multi_prompt:
            raise ValueError(
                "Kling 3.0 multi-shot needs at least one valid line."
            )
        if len(multi_prompt) > 5:
            raise ValueError(
                f"Kling 3.0 supports max 5 shots; got {len(multi_prompt)}."
            )
        total_duration = sum(s["duration"] for s in multi_prompt)
        if not 3 <= total_duration <= 15:
            raise ValueError(
                f"Total shot duration must be 3-15s; got {total_duration}s."
            )

        body: dict[str, Any] = {
            "prompt": "",
            "mode": kwargs["mode"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": str(total_duration),
            "sound": True,
            "multi_shots": True,
            "multi_prompt": multi_prompt,
        }
        first = (kwargs.get("image_url") or "").strip()
        if first:
            body["image_urls"] = [first]
        return body


# ============================================================ Motion control

class _KlingMotionControlBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Kling motion-control nodes.

    Motion control transfers the motion from a reference video onto a
    character in a reference image. Both inputs are required.
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "input_url": ("STRING", {
                    "default": "",
                    "tooltip": "Image URL of the subject (head/shoulders/torso). "
                               "JPEG/PNG, max 10MB, >340px, aspect ratio 2:5 to 5:2.",
                }),
                "video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Motion reference video URL. MP4/MOV, "
                               "3-30s, max 100MB.",
                }),
                "mode": (["std", "pro"], {
                    "default": "pro",
                    "tooltip": "std = 720p, pro = 1080p",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional guidance text (0-2500 chars).",
                }),
                "character_orientation": (["video", "image"], {
                    "default": "video",
                    "tooltip": "Where to read character orientation from.",
                }),
                "background_source": (["input_video", "input_image"], {
                    "default": "input_video",
                    "tooltip": "Where to read the background from.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        input_url = (kwargs.get("input_url") or "").strip()
        video_url = (kwargs.get("video_url") or "").strip()
        if not input_url:
            raise ValueError("Kling motion-control requires input_url.")
        if not video_url:
            raise ValueError("Kling motion-control requires video_url.")

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
    """Kling 3.0 motion-control — transfer motion onto a character."""
    MODEL = "kling-3.0/motion-control"


class Kling26MotionControl(_KlingMotionControlBase):
    """Kling 2.6 motion-control — cheaper alternative."""
    MODEL = "kling-2.6/motion-control"


# ============================================================ Kling 2.6

class Kling26T2V(BaseKieMarketVideoNode):
    """Kling 2.6 text-to-video.

    Pricing: $0.275/video 5s no audio, $0.55 with audio. Cheaper than 3.0
    and supports same durations.
    """

    MODEL = "kling-2.6/text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic establishing shot at golden hour.",
                }),
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
    """Kling 2.6 image-to-video. Requires image_url (passed as image_urls)."""

    MODEL = "kling-2.6/image-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Input image URL (required).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Subtle animation, gentle motion, cinematic.",
                }),
                "duration": (["5", "10"], {"default": "5"}),
                "sound": ("BOOLEAN", {"default": False}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        image_url = (kwargs.get("image_url") or "").strip()
        if not image_url:
            raise ValueError("Kling 2.6 I2V requires image_url.")
        return {
            "prompt": kwargs["prompt"],
            "image_urls": [image_url],
            "duration": str(kwargs["duration"]),
            "sound": bool(kwargs.get("sound", False)),
        }


# ============================================================ Kling 2.5 Turbo Pro

class _Kling25TurboBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Kling 2.5 Turbo Pro."""

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic establishing shot at sunset.",
                }),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blur, distort, and low quality",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
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
    """Kling 2.5 Turbo Pro — text-to-video (~$0.21 per 5s)."""
    MODEL = "kling/v2-5-turbo-text-to-video-pro"


class Kling25TurboProI2V(_Kling25TurboBase):
    """Kling 2.5 Turbo Pro — image-to-video (~$0.21 per 5s)."""

    MODEL = "kling/v2-5-turbo-image-to-video-pro"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        schema["required"] = {
            "image_url": ("STRING", {
                "default": "",
                "tooltip": "First-frame image URL (required).",
            }),
            **schema["required"],
        }
        schema["optional"]["end_image_url"] = ("STRING", {
            "default": "",
            "tooltip": "Optional last-frame URL.",
        })
        return schema

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body = super().build_input(**kwargs)
        image_url = (kwargs.get("image_url") or "").strip()
        if not image_url:
            raise ValueError("Kling 2.5 Turbo Pro I2V requires image_url.")
        body["image_url"] = image_url
        end_url = (kwargs.get("end_image_url") or "").strip()
        if end_url:
            body["end_image_url"] = end_url
        return body


# ============================================================ Kling 2.1

class _Kling21ImageBase(BaseKieMarketVideoNode):
    """Shared scaffolding for Kling 2.1 image-to-video tiers.

    Kling 2.1 Standard/Pro require image_url. Master also has text-to-video.
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "First-frame image URL (required).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Subtle cinematic animation.",
                }),
                "duration": (["5", "10"], {"default": "5"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blur, distort, and low quality",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        image_url = (kwargs.get("image_url") or "").strip()
        if not image_url:
            raise ValueError(f"{type(self).__name__} requires image_url.")
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "image_url": image_url,
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
    """Kling 2.1 Standard image-to-video (~$0.125/5s)."""
    MODEL = "kling/v2-1-standard"


class Kling21Pro(_Kling21ImageBase):
    """Kling 2.1 Pro image-to-video (~$0.25/5s, higher quality than Standard)."""
    MODEL = "kling/v2-1-pro"


class Kling21MasterI2V(_Kling21ImageBase):
    """Kling 2.1 Master image-to-video (~$0.80/5s, top quality)."""
    MODEL = "kling/v2-1-master-image-to-video"


class Kling21MasterT2V(BaseKieMarketVideoNode):
    """Kling 2.1 Master text-to-video (~$0.80/5s)."""

    MODEL = "kling/v2-1-master-text-to-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A cinematic establishing shot at sunset.",
                }),
                "duration": (["5", "10"], {"default": "5"}),
                "aspect_ratio": (_ASPECT_RATIOS_3, {"default": "16:9"}),
            },
            "optional": {
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": "blur, distort, and low quality",
                }),
                "cfg_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.1,
                }),
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
    """Shared scaffolding for Kling AI Avatar (lip-sync) nodes.

    Avatar generates a talking-head video from a portrait image and an audio
    track. The audio drives lip-sync; the image defines the face/identity.
    Output duration is bounded by the audio (max 5 minutes per docs, but
    pricing is tiered for ≤15s).
    """

    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 900.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Avatar portrait image URL "
                               "(JPEG/PNG, max 10MB).",
                }),
                "audio_url": ("STRING", {
                    "default": "",
                    "tooltip": "Voice audio URL "
                               "(MP3/WAV/AAC/M4A/OGG, max 100MB, ≤5min).",
                }),
            },
            "optional": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional guidance text (max 5000 chars).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        image_url = (kwargs.get("image_url") or "").strip()
        audio_url = (kwargs.get("audio_url") or "").strip()
        if not image_url:
            raise ValueError(f"{type(self).__name__} requires image_url.")
        if not audio_url:
            raise ValueError(f"{type(self).__name__} requires audio_url.")
        return {
            "image_url": image_url,
            "audio_url": audio_url,
            "prompt": kwargs.get("prompt", "") or "",
        }


class KlingAvatarStandard(_KlingAvatarBase):
    """Kling AI Avatar Standard — 720p lip-sync, ~$0.04/s (up to 15s)."""
    MODEL = "kling/ai-avatar-standard"


class KlingAvatarPro(_KlingAvatarBase):
    """Kling AI Avatar Pro — 1080p lip-sync, ~$0.08/s (up to 15s)."""
    MODEL = "kling/ai-avatar-pro"


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    # 3.0
    "GenesisKieKling30": Kling30,
    "GenesisKieKling30MultiShot": Kling30MultiShot,
    "GenesisKieKling30MotionControl": Kling30MotionControl,
    # 2.6
    "GenesisKieKling26T2V": Kling26T2V,
    "GenesisKieKling26I2V": Kling26I2V,
    "GenesisKieKling26MotionControl": Kling26MotionControl,
    # 2.5 Turbo Pro
    "GenesisKieKling25TurboProT2V": Kling25TurboProT2V,
    "GenesisKieKling25TurboProI2V": Kling25TurboProI2V,
    # 2.1
    "GenesisKieKling21Standard": Kling21Standard,
    "GenesisKieKling21Pro": Kling21Pro,
    "GenesisKieKling21MasterT2V": Kling21MasterT2V,
    "GenesisKieKling21MasterI2V": Kling21MasterI2V,
    # Avatar
    "GenesisKieKlingAvatarStandard": KlingAvatarStandard,
    "GenesisKieKlingAvatarPro": KlingAvatarPro,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieKling30": "Kie — Kling 3.0",
    "GenesisKieKling30MultiShot": "Kie — Kling 3.0 (multi-shot)",
    "GenesisKieKling30MotionControl": "Kie — Kling 3.0 Motion Control",
    "GenesisKieKling26T2V": "Kie — Kling 2.6 (T2V)",
    "GenesisKieKling26I2V": "Kie — Kling 2.6 (I2V)",
    "GenesisKieKling26MotionControl": "Kie — Kling 2.6 Motion Control",
    "GenesisKieKling25TurboProT2V": "Kie — Kling 2.5 Turbo Pro (T2V)",
    "GenesisKieKling25TurboProI2V": "Kie — Kling 2.5 Turbo Pro (I2V)",
    "GenesisKieKling21Standard": "Kie — Kling 2.1 Standard",
    "GenesisKieKling21Pro": "Kie — Kling 2.1 Pro",
    "GenesisKieKling21MasterT2V": "Kie — Kling 2.1 Master (T2V)",
    "GenesisKieKling21MasterI2V": "Kie — Kling 2.1 Master (I2V)",
    "GenesisKieKlingAvatarStandard": "Kie — Kling AI Avatar Standard",
    "GenesisKieKlingAvatarPro": "Kie — Kling AI Avatar Pro",
}
