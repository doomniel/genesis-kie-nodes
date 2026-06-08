# v0.5.0 — Music / Audio catalog (133 nodes total)

Adds **20 music & audio nodes** covering ElevenLabs (Market) + the full Suno API surface (Dedicated). Total catalog grows to **133 nodes**.

## 🎵 Coverage (20 new nodes)

### ElevenLabs (Market, 4)
- Audio Isolation (stem extraction from full mix)
- TTS Multilingual v2 (29+ languages, premium quality)
- TTS Turbo 2.5 (faster + cheaper, 32 languages)
- Text-to-Dialogue v3 (multi-speaker conversational)

### Suno Music Generation (Dedicated, 11)
- Generate Music, Extend Music
- Upload And Cover, Upload And Extend
- Music Cover, Add Instrumental, Add Vocals
- Boost Style, Replace Section
- Generate Persona, Mashup

### Suno Utilities (Dedicated, 5)
- Generate Lyrics (text output)
- Get Timestamped Lyrics (synchronous endpoint)
- Convert to WAV (lossless audio)
- Vocal Removal (multi-stem output)
- Generate MIDI from Audio

## 🔌 6th endpoint pattern: Suno

- `POST /api/v1/generate*` (music ops) + `GET /api/v1/generate/record-info` (shared polling)
- Sister utility endpoints with their own polling URLs (`/lyrics`, `/wav`, `/vocal-removal`, `/midi`)
- String-based status enum: `SUCCESS` / `*_FAILED` variants
- Result shape: `response.sunoData[]` (array of variants — Suno typically returns 2-3 candidates per request)

`KieClient` now exposes 6 endpoint patterns. New base classes:
- `BaseKieSunoMusicNode` (triple output: `audio_path`, `audio_id`, `all_paths_csv`)
- `BaseKieSunoTextNode` (text output)
- `BaseKieSunoAudioUtilityNode` (single audio file output)
- `BaseKieSunoStemSeparationNode` (multi-stem output)

## 🔥 Why triple output for Suno?

The `audio_id` field is **critical** for chaining workflows:
- Generate Music → use `audio_id` in Extend Music
- Generate Music → use `audio_id` in Add Vocals
- Generate Music → use `audio_id` in Generate Persona / Music Cover / Replace Section

Without exposing it, multi-step Suno workflows would be impossible in ComfyUI.

## 🔧 Fixed (in-release patches)

- **Lyrics endpoint** corrected from `/api/v1/lyrics/generate` (404) to `/api/v1/lyrics`.
- **Lyrics response parser** updated to handle `response.data[].text` array shape (concatenates 2-3 variations with `=== Title ===` markers).
- **`callBackUrl` auto-injection** — Suno requires this field even when polling. Placeholder URL injected automatically; never invoked because we poll for results.

## 🧪 Smoke validated

- Suno Generate Lyrics → "Cogsworth's Grin" with proper `[Verse]` structure
- ElevenLabs TTS Multilingual v2 → 15.7s, .mp3 generated
- Suno Generate Music V3_5 → 132.9s, 2 variants + `audio_id` for chaining

**Total: 133 nodes (70 video + 43 image + 20 music).**
