"""Suno utility nodes (lyrics, WAV conversion, vocal removal, MIDI).

5 utility endpoints in the Suno API family:

- Generate Lyrics              POST /api/v1/lyrics/generate
                                GET  /api/v1/lyrics/record-info
  Output: text (lyrics)

- Get Timestamped Lyrics       POST /api/v1/generate/get-timestamped-lyrics
                                (SYNCHRONOUS — returns immediately)
  Output: timestamped lyrics text

- Convert to WAV Format        POST /api/v1/wav/generate
                                GET  /api/v1/wav/record-info
  Output: WAV audio file path

- Vocal Removal / Stem Sep     POST /api/v1/vocal-removal/generate
                                GET  /api/v1/vocal-removal/record-info
  Output: multi-stem audio paths (vocals + instrumental)

- Generate MIDI from Audio     POST /api/v1/midi/generate
                                GET  /api/v1/midi/record-info
  Output: MIDI file path

Note: Get Timestamped Lyrics is the only synchronous endpoint —
it returns lyrics text directly without polling.
"""

from __future__ import annotations

from typing import Any

from ..base import (
    BaseKieNode,
    BaseKieSunoAudioUtilityNode,
    BaseKieSunoStemSeparationNode,
    BaseKieSunoTextNode,
    CATEGORY_MUSIC,
)
from ...client import KieClient, KieError


class SunoGenerateLyrics(BaseKieSunoTextNode):
    """Generate song lyrics from a creative prompt (Suno's lyrics API).

    Returns full lyrics as text. Use the output as input to Suno
    Generate Music in customMode=true to control the song's lyrics.

    Per docs.kie.ai cURL: endpoint is /api/v1/lyrics (NOT /lyrics/generate).
    Response shape: ``data.response.data[]`` array of objects with
    ``{text, title, status, errorMessage}``.
    """

    MODEL = "suno-lyrics"
    CREATE_ENDPOINT = "/api/v1/lyrics"
    POLLING_ENDPOINT = "/api/v1/lyrics/record-info"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": (
                        "A nostalgic song about late-night drives through "
                        "an empty city, feeling reflective and slightly hopeful."
                    ),
                    "tooltip": "Max 200 characters per docs.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        prompt = (kwargs.get("prompt") or "").strip()
        if not prompt:
            raise ValueError("Suno Generate Lyrics requires non-empty prompt.")
        if len(prompt) > 200:
            raise ValueError(
                f"Suno Generate Lyrics: prompt max 200 chars, got {len(prompt)}."
            )
        return {"prompt": prompt}

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        """Lyrics response shape: ``response.data = [{text, title, ...}]``."""
        response = data.get("response") or {}
        items = response.get("data")
        if isinstance(items, list) and items:
            # Concatenate all lyric variations (Suno typically returns 2-3).
            texts: list[str] = []
            for item in items:
                if isinstance(item, dict):
                    text = item.get("text")
                    title = item.get("title", "")
                    if isinstance(text, str) and text:
                        if title:
                            texts.append(f"=== {title} ===\n{text}")
                        else:
                            texts.append(text)
            if texts:
                return ("\n\n".join(texts),)

        # Fallback to base implementation for other shapes.
        return super().extract_output(data)


class SunoGetTimestampedLyrics(BaseKieNode):
    """Get timestamped lyrics for a Suno-generated track (SYNCHRONOUS).

    Per docs: unlike other Suno endpoints, this one returns immediately
    (no taskId polling). Output is timestamps + lyrics text.

    Use the audio_id from a Generate Music task to retrieve synced lyrics
    for karaoke / lyric video workflows.
    """

    MODEL = "suno-timestamped-lyrics"
    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("timestamped_lyrics",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "task_id from Suno Generate Music.",
                }),
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id within that task.",
                }),
            },
        }

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        task_id = (kwargs.get("task_id") or "").strip()
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not task_id or not audio_id:
            raise ValueError(
                "Suno Timestamped Lyrics requires both task_id and audio_id."
            )

        body = {"taskId": task_id, "audioId": audio_id}

        with KieClient() as client:
            # Direct POST without polling (synchronous endpoint).
            payload = client._request(
                "POST",
                "/api/v1/generate/get-timestamped-lyrics",
                json=body,
            )
        data = payload.get("data") or {}

        # The response shape can vary — try common keys.
        import json
        for key in ("lyricsData", "alignedWords", "timestampedLyrics", "lyrics"):
            value = data.get(key)
            if isinstance(value, str) and value:
                return (value,)
            if isinstance(value, (list, dict)):
                return (json.dumps(value, ensure_ascii=False, indent=2),)

        # Fallback: dump full data dict.
        return (json.dumps(data, ensure_ascii=False, indent=2),)


class SunoConvertToWAV(BaseKieSunoAudioUtilityNode):
    """Convert a Suno-generated track to high-quality WAV format.

    Use this when you need uncompressed audio for professional editing
    (DAW import, mastering, etc). MP3 output is lossy; WAV preserves
    full quality.

    Per docs cURL: body is { taskId, audioId, callBackUrl }.
    """

    MODEL = "suno-wav"
    CREATE_ENDPOINT = "/api/v1/wav/generate"
    POLLING_ENDPOINT = "/api/v1/wav/record-info"
    OUTPUT_EXT = "wav"
    RESULT_FIELD = "audioWavUrl"  # Best guess based on naming convention

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "task_id from a Suno music task.",
                }),
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id within that task.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        task_id = (kwargs.get("task_id") or "").strip()
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not task_id or not audio_id:
            raise ValueError("Suno WAV Convert requires both task_id and audio_id.")
        return {"taskId": task_id, "audioId": audio_id}


class SunoVocalRemoval(BaseKieSunoStemSeparationNode):
    """Separate vocals from instrumental on a Suno-generated track.

    Returns TWO outputs: vocals stem and instrumental stem.
    Also returns all_stems_csv for any additional stems Suno produces.

    Per docs cURL: type can be 'separate_vocal' (vocals + instrumental)
    or potentially other modes (full stem separation per instrument).
    """

    MODEL = "suno-vocal-removal"

    # Inherits CREATE_ENDPOINT / POLLING_ENDPOINT from base.

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "task_id from a Suno music task.",
                }),
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id within that task.",
                }),
                "type": (["separate_vocal", "stem"], {
                    "default": "separate_vocal",
                    "tooltip": (
                        "separate_vocal = vocals + instrumental (2 stems). "
                        "stem = full per-instrument breakdown if supported."
                    ),
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        task_id = (kwargs.get("task_id") or "").strip()
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not task_id or not audio_id:
            raise ValueError("Suno Vocal Removal requires both task_id and audio_id.")
        return {
            "taskId": task_id,
            "audioId": audio_id,
            "type": kwargs["type"],
        }


class SunoGenerateMIDI(BaseKieSunoAudioUtilityNode):
    """Generate a MIDI file from a Suno-generated audio track.

    Useful for: extracting melody, importing into a DAW, building
    sheet music, training other models on the structure.

    Per docs cURL: body is { taskId, audioId, callBackUrl }.
    """

    MODEL = "suno-midi"
    CREATE_ENDPOINT = "/api/v1/midi/generate"
    POLLING_ENDPOINT = "/api/v1/midi/record-info"
    OUTPUT_EXT = "mid"
    RESULT_FIELD = "midiUrl"

    RETURN_NAMES = ("midi_path",)

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "task_id from a Suno music task.",
                }),
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id within that task.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        task_id = (kwargs.get("task_id") or "").strip()
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not task_id or not audio_id:
            raise ValueError("Suno MIDI Generate requires both task_id and audio_id.")
        return {"taskId": task_id, "audioId": audio_id}


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSunoGenerateLyrics": SunoGenerateLyrics,
    "GenesisKieSunoGetTimestampedLyrics": SunoGetTimestampedLyrics,
    "GenesisKieSunoConvertToWAV": SunoConvertToWAV,
    "GenesisKieSunoVocalRemoval": SunoVocalRemoval,
    "GenesisKieSunoGenerateMIDI": SunoGenerateMIDI,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSunoGenerateLyrics": "Kie — Suno Generate Lyrics",
    "GenesisKieSunoGetTimestampedLyrics": "Kie — Suno Timestamped Lyrics",
    "GenesisKieSunoConvertToWAV": "Kie — Suno Convert to WAV",
    "GenesisKieSunoVocalRemoval": "Kie — Suno Vocal Removal (Stems)",
    "GenesisKieSunoGenerateMIDI": "Kie — Suno Generate MIDI",
}
