"""Google Gemini Omni multimodal video generation nodes (via Kie.ai).

Gemini Omni is Google's multimodal creation model — accepts text, images,
reference videos, reusable characters, and cloned voices in a single
generation call.

Slot system: each video has up to 7 reference slots:
- Each image: 1 slot
- Each video: 2 slots
- Each character_id: 1 slot
- Plus 1 audio_id (separate budget)

Covers all 3 Gemini Omni endpoints in Kie.ai:

- **Gemini Omni Video** (Market): POST /api/v1/jobs/createTask
  Model: ``gemini-omni-video``
  Multi-modal video generation with image_urls + audio_ids + video_list +
  character_id_list.

- **Gemini Omni Audio** (dedicated): POST /api/v1/omni/audio/create
  Creates a reusable voice (synchronous, returns kieAudioId).

- **Gemini Omni Character** (dedicated): POST /api/v1/omni/character/create
  Creates a reusable character (synchronous, returns kieCharacterId).

The audio + character endpoints are **synchronous** — they don't go through
the createTask + polling cycle. The video endpoint is the standard async
Market flow.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode, BaseKieNode, CATEGORY_VIDEO
from ...client import KieClient, KieError

log = logging.getLogger("genesis_kie")


# ============================================================ Gemini Omni Video

class GeminiOmniVideo(BaseKieMarketVideoNode):
    """Gemini Omni Video — multimodal video generation.

    Per docs cURL example:
        {
          "model": "gemini-omni-video",
          "input": {
            "prompt": "Create a futuristic night city short film...",
            "image_urls": ["https://.../scene-1.png", "https://.../scene-2.png"],
            "audio_ids": ["audio_01hx8p0demo"],
            "video_list": [{"url": "https://.../source-video.mp4", "start": 0, "ends": 10}],
            "duration": "4"
          }
        }

    Slot budget per video (7 total):
    - Each image_url: 1 slot
    - Each video in video_list: 2 slots
    - Each character_id: 1 slot
    - audio_ids: max 1 ID, separate budget

    When video input is provided, output duration is determined by the
    model automatically (the ``duration`` parameter is ignored).
    """

    MODEL = "gemini-omni-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1500.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Create a cinematic short film with a slow push-in camera.",
                }),
                "duration": (["4", "6", "8", "10"], {
                    "default": "4",
                    "tooltip": "Output duration (ignored if video input provided).",
                }),
            },
            "optional": {
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs (1 slot each, max 7 combined).",
                }),
                "video_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional reference video URL (uses 2 slots).",
                }),
                "video_start": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Video reference start time (seconds).",
                }),
                "video_ends": ("FLOAT", {
                    "default": 10.0, "min": 0.0, "max": 600.0, "step": 0.1,
                    "tooltip": "Video reference end time (seconds).",
                }),
                "audio_ids": ("STRING", {
                    "default": "",
                    "tooltip": "Single audio_id from a Gemini Omni Audio creation call.",
                }),
                "character_ids": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated character IDs (1 slot each, max 3).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
        }

        imgs = self._csv((kwargs.get("image_urls") or "").strip())
        video_url = (kwargs.get("video_url") or "").strip()
        chars = self._csv((kwargs.get("character_ids") or "").strip())

        # Validate slot budget: 7 total.
        slots = len(imgs) + (2 if video_url else 0) + len(chars)
        if slots > 7:
            raise ValueError(
                f"Gemini Omni: slot budget exceeded ({slots}/7). "
                f"Each image=1, each video=2, each character=1."
            )
        if len(chars) > 3:
            raise ValueError(f"Gemini Omni: max 3 characters, got {len(chars)}.")

        if imgs:
            body["image_urls"] = imgs
        if video_url:
            body["video_list"] = [{
                "url": video_url,
                "start": float(kwargs.get("video_start", 0.0)),
                "ends": float(kwargs.get("video_ends", 10.0)),
            }]
        if chars:
            body["character_id_list"] = chars

        audio_ids = self._csv((kwargs.get("audio_ids") or "").strip())
        if audio_ids:
            if len(audio_ids) > 1:
                raise ValueError(f"Gemini Omni: max 1 audio_id, got {len(audio_ids)}.")
            body["audio_ids"] = audio_ids

        return body

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Gemini Omni Audio

class GeminiOmniAudioCreate(BaseKieNode):
    """Gemini Omni Audio — create a reusable voice.

    Per docs cURL example (dedicated endpoint, NOT createTask):
        POST /api/v1/omni/audio/create
        {
          "audio_id": "achernar",
          "name": "achernar Narrator",
          "voice_description": "A calm, clear, and friendly male voice...",
          "example_dialogue": "Hello, I am achernar"
        }

    Returns a ``kieAudioId`` that can then be passed to
    :class:`GeminiOmniVideo` via the ``audio_ids`` field.

    This is a **synchronous** call — it doesn't poll a task. The node
    returns the kieAudioId as a string so it can be piped into a Gemini
    Omni Video node downstream.
    """

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kie_audio_id",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_id": ("STRING", {
                    "default": "achernar",
                    "tooltip": "Source voice preset name (e.g. 'achernar').",
                }),
                "name": ("STRING", {
                    "default": "achernar Narrator",
                    "tooltip": "Friendly name for this voice.",
                }),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "A calm, clear, and friendly male voice suitable for tech explainers and daily conversation.",
                }),
            },
            "optional": {
                "example_dialogue": ("STRING", {
                    "multiline": True,
                    "default": "Hello, I am achernar.",
                    "tooltip": "Optional example dialogue to seed voice cloning.",
                }),
            },
        }

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        with KieClient() as client:
            data = client.create_omni_audio(
                audio_id=kwargs["audio_id"],
                name=kwargs["name"],
                voice_description=kwargs["voice_description"],
                example_dialogue=(kwargs.get("example_dialogue") or "").strip(),
            )

        kie_audio_id = data.get("kieAudioId")
        if not kie_audio_id:
            raise KieError(f"Gemini Omni Audio create did not return kieAudioId. Response: {data}")
        log.info("Gemini Omni Audio created kieAudioId=%s", kie_audio_id)
        return (kie_audio_id,)


# ============================================================ Gemini Omni Character

class GeminiOmniCharacterCreate(BaseKieNode):
    """Gemini Omni Character — create a reusable character.

    Per Kie.ai docs (dedicated endpoint, parallel to audio create):
        POST /api/v1/omni/character/create
        {
          "character_id": "<source_template>",
          "name": "<friendly name>",
          "character_description": "<...>",
          "image_urls": ["https://..."]
        }

    Returns a character_id that can then be passed to
    :class:`GeminiOmniVideo` via the ``character_ids`` field.

    **Note**: The exact endpoint body shape was inferred from the audio
    endpoint pattern. If the actual API uses a different field layout,
    adjust :meth:`KieClient.create_omni_character` accordingly.
    """

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kie_character_id",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "character_id": ("STRING", {
                    "default": "",
                    "tooltip": "Source character template identifier.",
                }),
                "name": ("STRING", {
                    "default": "my_character_01",
                    "tooltip": "Friendly name for this character.",
                }),
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "A friendly cartoon character with expressive eyes and fluid movements.",
                }),
                "image_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated reference image URLs of the character.",
                }),
            },
        }

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        cid = (kwargs.get("character_id") or "").strip()
        if not cid:
            raise ValueError("Gemini Omni Character requires character_id.")
        imgs = self._csv((kwargs.get("image_urls") or "").strip())
        if not imgs:
            raise ValueError("Gemini Omni Character requires at least one image_url.")

        with KieClient() as client:
            data = client.create_omni_character(
                character_id=cid,
                name=kwargs["name"],
                character_description=kwargs["character_description"],
                image_urls=imgs,
            )

        # Try a few common naming conventions for the returned ID.
        kie_char_id = (
            data.get("kieCharacterId")
            or data.get("character_id")
            or data.get("characterId")
        )
        if not kie_char_id:
            raise KieError(
                f"Gemini Omni Character create did not return a character ID. "
                f"Response: {data}"
            )
        log.info("Gemini Omni Character created id=%s", kie_char_id)
        return (kie_char_id,)

    @staticmethod
    def _csv(value: str) -> list[str]:
        if not value:
            return []
        return [s.strip() for s in value.split(",") if s.strip()]


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGeminiOmniVideo": GeminiOmniVideo,
    "GenesisKieGeminiOmniAudioCreate": GeminiOmniAudioCreate,
    "GenesisKieGeminiOmniCharacterCreate": GeminiOmniCharacterCreate,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGeminiOmniVideo": "Kie — Gemini Omni Video",
    "GenesisKieGeminiOmniAudioCreate": "Kie — Gemini Omni Audio (create voice)",
    "GenesisKieGeminiOmniCharacterCreate": "Kie — Gemini Omni Character (create)",
}
