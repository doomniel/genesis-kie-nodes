"""Kling video generation nodes.

This batch covers the two highest-impact Kling models:

- **Kling 3.0** — flagship with multi-shot, element references, modes
- **Kling 2.5 Turbo Pro** — production workhorse (t2v + i2v)

Both use the Market endpoint ``/api/v1/jobs/createTask``.

Pricing reference (Kie.ai, 2026):
    Kling 3.0 std 720p          $0.07/s no audio, $0.10/s w/audio
    Kling 3.0 pro 1080p         $0.09/s no audio, $0.135/s w/audio
    Kling 3.0 4K                $0.335/s
    Kling 2.5 Turbo Pro 5s      $0.21/video
    Kling 2.5 Turbo Pro 10s     $0.42/video
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode


# Common enums.
_ASPECT_RATIOS = ["16:9", "9:16", "1:1"]


# ============================================================ Kling 3.0

class _Kling30Base(BaseKieMarketVideoNode):
    """Shared scaffolding for Kling 3.0 single-shot generation.

    Multi-shot mode is exposed via :class:`Kling30MultiShot` to keep the UI
    of the basic node simple.
    """

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
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
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

    Accepts a compact "scene script" string with one shot per line, formatted
    as ``<duration>:<prompt>``. Example::

        3:A happy dog running in the park
        4:The dog jumps over a fence
        3:Close-up of the dog's smiling face

    Total duration is the sum of per-shot durations (must equal a valid
    Kling duration: 3..15 seconds).
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
                    "tooltip": "One shot per line, formatted as <duration>:<prompt>. "
                               "Max 5 shots, each 1-12s, total 3-15s.",
                }),
                "mode": (["std", "pro", "4K"], {"default": "pro"}),
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
            },
            "optional": {
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional first-frame image (only first frame "
                               "supported in multi-shot mode).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        script: str = kwargs["shots_script"]

        multi_prompt: list[dict[str, Any]] = []
        for line in script.splitlines():
            line = line.strip()
            if not line:
                continue
            if ":" not in line:
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
                "Kling 3.0 multi-shot needs at least one valid line of "
                "<duration>:<prompt> in shots_script."
            )
        if len(multi_prompt) > 5:
            raise ValueError("Kling 3.0 supports at most 5 shots; got %d."
                             % len(multi_prompt))

        total_duration = sum(s["duration"] for s in multi_prompt)
        if not 3 <= total_duration <= 15:
            raise ValueError(
                f"Total shot duration must be 3-15s; got {total_duration}s."
            )

        body: dict[str, Any] = {
            "prompt": "",  # required field but unused in multi-shot mode
            "mode": kwargs["mode"],
            "aspect_ratio": kwargs["aspect_ratio"],
            "duration": str(total_duration),
            "sound": True,  # always-on in multi-shot
            "multi_shots": True,
            "multi_prompt": multi_prompt,
        }

        first = (kwargs.get("image_url") or "").strip()
        if first:
            body["image_urls"] = [first]

        return body


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
                "aspect_ratio": (_ASPECT_RATIOS, {"default": "16:9"}),
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
    """Kling 2.5 Turbo Pro — image-to-video (~$0.21 per 5s).

    Requires ``image_url`` (first frame) to drive the animation.
    """

    MODEL = "kling/v2-5-turbo-image-to-video-pro"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        schema = super().INPUT_TYPES()
        # Add image_url as required at the top of required.
        schema["required"] = {
            "image_url": ("STRING", {
                "default": "",
                "tooltip": "First-frame image URL (required).",
            }),
            **schema["required"],
        }
        # Also accept optional last frame.
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


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieKling30": Kling30,
    "GenesisKieKling30MultiShot": Kling30MultiShot,
    "GenesisKieKling25TurboProT2V": Kling25TurboProT2V,
    "GenesisKieKling25TurboProI2V": Kling25TurboProI2V,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieKling30": "Kie — Kling 3.0",
    "GenesisKieKling30MultiShot": "Kie — Kling 3.0 (multi-shot)",
    "GenesisKieKling25TurboProT2V": "Kie — Kling 2.5 Turbo Pro (T2V)",
    "GenesisKieKling25TurboProI2V": "Kie — Kling 2.5 Turbo Pro (I2V)",
}
