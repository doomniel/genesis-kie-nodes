"""ElevenLabs audio nodes (via GenesisLab proxy / Kie.ai Market endpoint).

Covers all 4 ElevenLabs endpoints:

- elevenlabs/audio-isolation                  (stem extraction from full mix)
- elevenlabs/text-to-dialogue-v3              (multi-speaker dialogue TTS)
- elevenlabs/text-to-speech-multilingual-v2   (TTS, 29+ languages, premium)
- elevenlabs/text-to-speech-turbo-2-5         (TTS, faster + cheaper)

AudioIsolation accepts a native ComfyUI AUDIO input (dict with waveform
+ sample_rate). The 3 TTS nodes are text-only and don't need audio input.
"""

from __future__ import annotations

from typing import Any

from ..base import BaseKieMarketAudioNode
from ...client.upload import upload_audio


_EL_VOICE_PRESETS = [
    "Rachel", "Adam", "Antoni", "Bella", "Sam",
    "Domi", "Elli", "Josh", "Arnold",
    "custom",
]
_EL_OUTPUT_FORMATS = ["mp3_44100_128", "mp3_44100_192", "pcm_22050", "pcm_44100"]


class ElevenLabsAudioIsolation(BaseKieMarketAudioNode):
    """ElevenLabs Audio Isolation — extracts clean vocal stem from a mix.

    Accepts a native ComfyUI AUDIO input. The waveform is uploaded to
    GenesisLab temp storage as WAV and the resulting URL is sent to
    ElevenLabs' stem-separation model.
    """

    MODEL = "elevenlabs/audio-isolation"
    POLL_INTERVAL_SECONDS = 2.0
    TIMEOUT_SECONDS = 180.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio": ("AUDIO", {
                    "tooltip": "Source audio (mix). Will isolate vocals.",
                }),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        audio = kwargs.get("audio")
        if audio is None:
            raise ValueError("ElevenLabs Audio Isolation requires audio input.")
        url = upload_audio(audio)
        return {"audio_url": url}


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
                }),
                "similarity_boost": ("FLOAT", {
                    "default": 0.75, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "style": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
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
    """ElevenLabs Multilingual v2 — 29+ languages, premium quality."""
    MODEL = "elevenlabs/text-to-speech-multilingual-v2"


class ElevenLabsTTSTurbo25(_ElevenLabsTTSBase):
    """ElevenLabs Turbo 2.5 — faster + cheaper, 32 languages."""
    MODEL = "elevenlabs/text-to-speech-turbo-2-5"


class ElevenLabsTextToDialogueV3(BaseKieMarketAudioNode):
    """ElevenLabs Text-to-Dialogue v3 — multi-speaker conversational audio."""

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


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieElevenLabsAudioIsolation": ElevenLabsAudioIsolation,
    "GenesisKieElevenLabsTTSMultilingualV2": ElevenLabsTTSMultilingualV2,
    "GenesisKieElevenLabsTTSTurbo25": ElevenLabsTTSTurbo25,
    "GenesisKieElevenLabsTextToDialogueV3": ElevenLabsTextToDialogueV3,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieElevenLabsAudioIsolation": "ElevenLabs Audio Isolation",
    "GenesisKieElevenLabsTTSMultilingualV2": "ElevenLabs TTS Multilingual v2",
    "GenesisKieElevenLabsTTSTurbo25": "ElevenLabs TTS Turbo 2.5",
    "GenesisKieElevenLabsTextToDialogueV3": "ElevenLabs Text-to-Dialogue v3",
}
