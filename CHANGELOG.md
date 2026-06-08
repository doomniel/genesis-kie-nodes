# Changelog

All notable changes to **genesis-kie-nodes** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Planned for v0.7.0
- Music Video Generation (Suno-related)
- Sounds Generation endpoint
- Gemini Omni Character (multimodal)

### Planned for v0.8.0+
- Example `.json` workflows
- `KieCreditsCheck` node (balance preview)
- Production deploy guide for GenesisLab

---

## [0.6.0] — 2026-06-07

### Added
- **17 LLM / Chat nodes** across 4 families × 3 endpoint sub-patterns:
  - **GPT (3):** GPT 5.2 (Chat Completions), GPT 5.4 + 5.5 (Responses API)
  - **Claude (7):** Opus 4.5/4.6/4.7/4.8, Sonnet 4.5/4.6, Haiku 4.5
  - **Codex (1):** GPT Codex with version dropdown (5 variants: 5/5.1/5.2/5.3/5.4-codex)
  - **Gemini (6):** 2.5 Pro/Flash, 3 Pro/Flash, 3.1 Pro, 3.5 Flash (all OpenAI-compat)
- **7th endpoint pattern** in `KieClient`: `chat_completion()` (first synchronous endpoint — no taskId, no polling, 120s timeout).
- **4 new base classes** in `nodes/base.py`:
  - `BaseKieChatNode` (abstract)
  - `BaseKieChatOpenAINode` (OpenAI Chat Completions shape)
  - `BaseKieChatResponsesNode` (Responses API + adjustable `reasoning_effort`)
  - `BaseKieChatAnthropicNode` (Anthropic Messages + optional `thinking` blocks)
- **Multimodal input** (`image_url`) for all 17 LLM nodes that support vision.
- Professional outputs: `(text, tokens_used)` — built-in cost tracking.
- `smoke_phase5.py` and `debug_codex.py` test runners.

### Fixed
- **Codex `max_output_tokens` 500 bug**: removed `max_output_tokens` from Responses API body. Kie.ai's gateway returns `{"code":500,"msg":"Server exception"}` when this field is present on the `/api/v1/responses` endpoint. GPT 5.5 tolerated it on its different endpoint, but the field is non-standard in the Responses API spec anyway. Verified via 3-variant debug script.
- **Gemini "model not supported" 422 bug**: `_GeminiChatBase` now strips `model` from the request body. Kie's Gemini gateway rejects bodies containing it because the endpoint path already identifies the model. GPT and Claude bases still include `model` as expected.

### Total
- **150 nodes** across 4 modalities and 7 endpoint patterns.

---

## [0.5.0] — 2026-06-07

### Added
- **20 music / audio nodes** across 2 families × 3 endpoint sub-patterns:
  - **ElevenLabs (Market, 4):** Audio Isolation, TTS Multilingual v2, TTS Turbo 2.5, Text-to-Dialogue v3
  - **Suno Music Generation (Dedicated, 11):** Generate, Extend, UploadAndCover, UploadAndExtend, MusicCover, AddInstrumental, AddVocals, BoostStyle, ReplaceSection, GeneratePersona, Mashup
  - **Suno Utilities (Dedicated, 5):** GenerateLyrics, TimestampedLyrics (synchronous), ConvertToWAV, VocalRemoval (stem separation), GenerateMIDI
- **6th endpoint pattern** in `KieClient`: `run_suno_task()` for Suno's dedicated endpoints (string-based status enum: SUCCESS, *_FAILED variants).
- **4 new base classes** in `nodes/base.py`:
  - `BaseKieSunoMusicNode` (3 outputs: `audio_path`, `audio_id`, `all_paths_csv` — `audio_id` enables chaining to Extend/Cover/AddVocals)
  - `BaseKieSunoTextNode` (text output for lyrics)
  - `BaseKieSunoAudioUtilityNode` (single audio file output for WAV/MIDI)
  - `BaseKieSunoStemSeparationNode` (multi-stem output for vocal removal)
- Auto-injection of `callBackUrl` placeholder when not provided (Suno requires the field even when polling).

### Fixed
- **Lyrics endpoint path**: corrected `/api/v1/lyrics/generate` → `/api/v1/lyrics` (per docs.kie.ai cURL).
- **Lyrics response extraction**: override `extract_output()` to parse `response.data[].text` array shape (concatenates 2-3 lyric variations with `=== Title ===` markers).

### Total
- **133 nodes** (70 video + 43 image + 20 music).

---

## [0.4.0] — 2026-06-07

### Added
- **43 image nodes** across 11 families:
  - Seedream 4.5 (7): T2I, I2I, edit, multi-image variants
  - Google (7): Imagen, Nano Banana variants
  - Ideogram 3 (6): T2I, edit, magic prompt
  - Qwen-Image 2512 (5): T2I, edit
  - Flux 2.0 (4): dev, schnell, pro, ultra
  - GPT Image 2.0 (4): variants
  - Image Utilities (3): Topaz Upscale, BG remove, restore
  - Wan 2.5 Image (2)
  - Grok Imagine (2)
  - GPT 4o Image dedicated (1)
  - Flux Kontext Pro/Max dedicated (1)
  - Z-Image Turbo (1)
- **3 new endpoint patterns** in `KieClient`:
  - 4th: `run_4o_image()` (GPT 4o Image dedicated, Veo-style polling)
  - 5th: `run_flux_kontext()` (Flux Kontext dedicated)
  - Several new utility nodes for image post-processing.
- Output: `(IMAGE,)` tensor B×H×W×C for direct Save/Preview node compatibility.

### Fixed
- Flux Kontext response parsing: handle both `resultUrls` (array) and `resultImageUrl` (singular) shapes.

### Total
- **113 nodes** (70 video + 43 image).

---

## [0.3.0] — 2026-06-07

### Added
- **70 video nodes** across 12 families — full coverage of Kie's video Market + dedicated endpoints:
  - Wan (16): T2V, I2V, image-ref multi for Wan 2.2 A14B and Wan 2.5
  - Kling (14): 2.5 Pro/Standard with T2V, I2V, lipsync, image-ref variants
  - Sora (8): Sora 2 / Sora 2 Pro variants (note: API sunset planned Sep 2026)
  - Bytedance (8): Seedance, Seedance Lite, Seedance Pro
  - Hailuo (5): Hailuo 02, Hailuo 02 Pro (lipsync, I2V)
  - HappyHorse (4): First/Last Frame, Multi-image, Lipsync
  - Grok (4): video variants
  - Veo dedicated (3): Veo 3, 3.1 Fast, 3.1 Pro
  - Runway dedicated (3): Gen-4 Turbo, Gen-4 Aleph variants
  - Gemini (3): video variants
  - Topaz (1): Video Upscale
  - Infinitalk (1)
- **3 endpoint patterns** in `KieClient`:
  - 1st: `run_market()` (generic `/api/v1/jobs/createTask` for most families)
  - 2nd: `run_veo()` (Veo dedicated, `successFlag`-based polling)
  - 3rd: `run_runway()` (Runway dedicated, status-string polling)
- Frame extraction pipeline: download MP4 → decode → tensor B×H×W×C with fps + frame_count outputs.

### Total
- **70 video nodes**, baseline release.

---

[Unreleased]: https://github.com/doomniel/genesis-kie-nodes/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.6.0
[0.5.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.5.0
[0.4.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.4.0
[0.3.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.3.0
