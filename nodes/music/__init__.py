"""Music / audio generation nodes for Genesis-Kie.

20 music nodes across 2 patterns:

  ElevenLabs (Market, 4):
    - Audio Isolation (stem extraction)
    - Text-to-Dialogue v3 (multi-speaker conversational TTS)
    - TTS Multilingual v2 (premium TTS, 29+ languages)
    - TTS Turbo 2.5 (fast TTS, 32 languages)

  Suno (Dedicated, 16):
    Music Generation (11):
    - Generate Music, Extend Music
    - Upload And Cover, Upload And Extend
    - Music Cover, Add Instrumental, Add Vocals
    - Boost Style, Replace Section
    - Generate Persona, Mashup

    Utilities (5):
    - Generate Lyrics (text output)
    - Get Timestamped Lyrics (text output, synchronous)
    - Convert to WAV (audio output, lossless)
    - Vocal Removal (multi-stem output)
    - Generate MIDI (MIDI file output)

Total: 4 ElevenLabs + 16 Suno = 20 music nodes.
"""

from __future__ import annotations

from . import elevenlabs, suno_music, suno_utils

NODE_CLASS_MAPPINGS: dict[str, type] = {}
NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {}

for module in (elevenlabs, suno_music, suno_utils):
    NODE_CLASS_MAPPINGS.update(getattr(module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(
        getattr(module, "NODE_DISPLAY_NAME_MAPPINGS", {})
    )
