"""ElevenLabs audio nodes (via Kie.ai's Market endpoint).

Covers all 4 ElevenLabs endpoints in Kie.ai:

- elevenlabs/audio-isolation                  (stem extraction from full mix)
- elevenlabs/text-to-dialogue-v3              (TTS multi-speaker dialogue)
- elevenlabs/text-to-speech-multilingual-v2   (TTS, 29+ languages, premium)
- elevenlabs/text-to-speech-turbo-2-5         (TTS, faster + cheaper, 32 langs)

All use the generic Market pattern: POST /api/v1/jobs/createTask with
``model`` + ``input`` body. Result audio URL lives in ``data._parsed_result.resultUrls``.

ElevenLabs voices: each TTS model accepts a ``voice`` field. Common
preset voices include "Rachel", "Antoni", "Bella", "Sam", "Adam".
Custom voice IDs (sha-256 hash) also accepted.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketAudioNode


_EL_VOICE_PRESETS = [
    "Rachel", "Adam", "Antoni", "Bella", "Sam",
    "Domi", "Elli", "Josh", "Arnold",
    "custom",
]
_EL_OUTPUT_FORMATS = ["mp3_44100_128", "mp3_44100_192", "pcm_22050", "pcm_44100"]


class ElevenLabsAudioIsolation(BaseKieMarketAudioNode):
    """ElevenLabs Audio Isolation — extracts clean vocal stem from a mix.

    Per kie.ai docs: removes background music/noise and isolates vocals
    using ElevenLabs' proprietary stem-separation model. Single audio URL
    input, single (vocals-only) audio output.
    """

    MODEL = "elevenlabs/audio-isolation"
    POLL_INTERVAL_SECONDS = 2.0
    TIMEOUT_SECONDS = 180.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_url": ("STRING", {
                    "default": "",
                    "tooltip": "Source audio URL (mp3/wav/ogg).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        audio = (kwargs.get("audio_url") or "").strip()
        if not audio:
            raise ValueError("ElevenLabs Audio Isolation requires audio_url.")
        return {"audio_url": audio}


class _ElevenLabsTTSBase(BaseKieMarketAudioNode):
    """Common scaffolding for ElevenLabs text-to-speech models."""

    POLL_INTERVAL_SECONDS = 2.0
    TIMEOUT_SECONDS = 180.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True,
                    "default": "Hello, this is a voice synthesis demo.",
                }),
                "voice": (_EL_VOICE_PRESETS, {
                    "default": "Rachel",
                    "tooltip": "Preset voice name or 'custom' to use voice_id field.",
                }),
            },
            "optional": {
                "voice_id": ("STRING", {
                    "default": "",
                    "tooltip": "Custom voice ID (used when voice='custom').",
                }),
                "output_format": (_EL_OUTPUT_FORMATS, {"default": "mp3_44100_128"}),
                "stability": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Voice stability (0=expressive, 1=monotone).",
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Voice similarity to reference (higher = more accurate).",
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Style exaggeration (v3+ models only).",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        text = (kwargs.get("text") or "").strip()
        if not text:
            raise ValueError(f"{type(self).__name__} requires non-empty text.")

        voice = kwargs.get("voice", "Rachel")
        if voice == "custom":
            voice = (kwargs.get("voice_id") or "").strip()
            if not voice:
                raise ValueError(
                    f"{type(self).__name__}: voice='custom' requires voice_id."
                )

        body: dict[str, Any] = {
            "text": text,
            "voice": voice,
            "output_format": kwargs.get("output_format", "mp3_44100_128"),
            "stability": float(kwargs.get("stability", 0.5)),
            "similarity_boost": float(kwargs.get("similarity_boost", 0.75)),
        }
        style = float(kwargs.get("style", 0.0))
        if style > 0:
            body["style"] = style
        return body


class ElevenLabsTTSMultilingualV2(_ElevenLabsTTSBase):
    """ElevenLabs Multilingual v2 — 29+ languages, premium quality.

    Use for: marketing voiceovers, audiobook narration, dubbing,
    localized content where voice quality matters most.
    """
    MODEL = "elevenlabs/text-to-speech-multilingual-v2"


class ElevenLabsTTSTurbo25(_ElevenLabsTTSBase):
    """ElevenLabs Turbo 2.5 — faster + cheaper, 32 languages.

    Use for: real-time TTS, AI agents, interactive applications
    where speed > absolute fidelity.
    """
    MODEL = "elevenlabs/text-to-speech-turbo-2-5"


class ElevenLabsTextToDialogueV3(BaseKieMarketAudioNode):
    """ElevenLabs Text-to-Dialogue v3 — multi-speaker conversational audio.

    Generate realistic back-and-forth dialogue between multiple voices.
    Input is structured text with speaker tags.

    Per docs: input format is text with [Speaker1] / [Speaker2] markers
    (or similar tagging) — the model parses speakers and assigns voices.
    """

    MODEL = "elevenlabs/text-to-dialogue-v3"
    POLL_INTERVAL_SECONDS = 3.0
    TIMEOUT_SECONDS = 240.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "dialogue": ("STRING", {
                    "multiline": True,
                    "default": (
                        "[Speaker1] Hey, did you finish the project?\n"
                        "[Speaker2] Yes, I just submitted it five minutes ago.\n"
                        "[Speaker1] Great work! Let's celebrate."
                    ),
                    "tooltip": "Multi-speaker text with [SpeakerN] tags.",
                }),
            },
            "optional": {
                "voices": ("STRING", {
                    "default": "Rachel,Adam",
                    "tooltip": "Comma-separated voice names, one per speaker.",
                }),
                "output_format": (_EL_OUTPUT_FORMATS, {"default": "mp3_44100_128"}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        dialogue = (kwargs.get("dialogue") or "").strip()
        if not dialogue:
            raise ValueError("ElevenLabs Text-to-Dialogue requires dialogue text.")

        voices_csv = (kwargs.get("voices") or "Rachel,Adam").strip()
        voices = [v.strip() for v in voices_csv.split(",") if v.strip()]

        return {
            "text": dialogue,
            "voices": voices,
            "output_format": kwargs.get("output_format", "mp3_44100_128"),
        }


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieElevenLabsAudioIsolation": ElevenLabsAudioIsolation,
    "GenesisKieElevenLabsTTSMultilingualV2": ElevenLabsTTSMultilingualV2,
    "GenesisKieElevenLabsTTSTurbo25": ElevenLabsTTSTurbo25,
    "GenesisKieElevenLabsTextToDialogueV3": ElevenLabsTextToDialogueV3,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieElevenLabsAudioIsolation": "Kie — ElevenLabs Audio Isolation",
    "GenesisKieElevenLabsTTSMultilingualV2": "Kie — ElevenLabs TTS Multilingual v2",
    "GenesisKieElevenLabsTTSTurbo25": "Kie — ElevenLabs TTS Turbo 2.5",
    "GenesisKieElevenLabsTextToDialogueV3": "Kie — ElevenLabs Text-to-Dialogue v3",
}
