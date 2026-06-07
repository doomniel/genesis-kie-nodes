# Changelog

All notable changes to `genesis-kie-nodes` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] — 2026-06-06

### Added — Complete Kie.ai video catalog (58 → 70 nodes)

**Runway dedicated API (+3, new module `nodes/video/runway.py`)**
- Runway Gen-4 Turbo (`POST /api/v1/runway/generate`)
- Runway Extend (`POST /api/v1/runway/extend`) — only 720p sources extendable per docs
- Runway Aleph V2V editing (`POST /api/v1/aleph/generate`) — 5s output cap
- `KieClient.run_runway(path, body)` for the dedicated pattern: camelCase
  body, no `input` wrapper, polled via `/runway/record-detail`, result
  extracted from `data.videoInfo.videoUrl`.

**Sora 2 extras (+4)**
- Sora Watermark Remover (model `sora-watermark-remover`, ~$0.05/video).
  Only accepts Sora.com URLs starting with `sora.chatgpt.com`.
- Sora 2 Characters (create reusable character from a video clip).
- Sora 2 Characters Pro (extract character from a prior task at given timestamps).
- Sora 2 Pro Storyboard (multi-scene up to 25s). **Body schema inferred**
  from product page description — module docstring documents this and
  points to docs.kie.ai for canonical spec when published.

**Sora 2 Pro corrections**
- Added required `size` parameter: `"standard"` (720p, $0.45/10s) or `"high"`
  (1080p HD, $1.00/10s). Previously missing from batch 2.
- `n_frames` now supports `"10"` or `"15"` (Pro tier).

**HappyHorse 1.0 family completed (+2)**
- Reference-to-Video (1-9 reference images, uses `character1`/`character2`
  conventions in prompt for ordered referencing).
- Video Edit (V2V with `audio_setting`: `"auto"` or `"origin"`, optional
  0-5 reference images).

**Gemini Omni multimodal (+3, new module `nodes/video/gemini_omni.py`)**
- Gemini Omni Video (model `gemini-omni-video`) — multimodal with slot
  budget: 7 total slots (image=1, video=2, character=1), audio=1 separate.
- Gemini Omni Audio create (`POST /api/v1/omni/audio/create`, synchronous,
  returns `kieAudioId`).
- Gemini Omni Character create (`POST /api/v1/omni/character/create`,
  synchronous, returns `kieCharacterId`). **Path inferred** from audio
  endpoint pattern.
- `KieClient.create_omni_audio()` + `create_omni_character()` for the
  sync helper endpoints.

### Changed
- `KieClient`: refactored to support three endpoint patterns explicitly
  (Market generic, Veo dedicated, Runway dedicated) plus the sync Omni
  helpers. Documented in the module docstring.
- `_OK_CODES` extended from `{200}` to `{0, 200}` — Gemini Omni Audio
  returns `code: 0` for success (not 200).
- `nodes/video/__init__.py` aggregates 11 sub-modules (was 9), adding
  `runway` and `gemini_omni`.
- `nodes/video/happyhorse.py` rewrites with all 4 modes (T2V/I2V/R2V/Edit)
  using the correct `happyhorse/*` endpoint slugs.
- `nodes/video/sora.py` expands from 4 to 8 classes; Pro tier now uses
  separate `_SoraProBase` with the `size` selector.

### Validated
- All 70 nodes import cleanly into a fresh ComfyUI install.
- Distribution by family: Wan 16 · Kling 14 · Sora 2 8 · Bytedance 8 ·
  Hailuo 5 · Grok 4 · HappyHorse 4 · Veo 3 · Runway 3 · Gemini Omni 3 ·
  Topaz 1 · Infinitalk 1.

### Known caveats
- **Sora 2 Pro Storyboard**: request body uses `scenes: [{duration, prompt,
  reference_image_url?}]` array — inferred from product page since the
  OpenAPI spec was not exposed in `docs.kie.ai/.md` form at release time.
  Will be updated once canonical spec is available.
- **Gemini Omni Character**: request path (`/api/v1/omni/character/create`)
  and body shape (`character_id`, `name`, `character_description`,
  `image_urls`) inferred from audio endpoint pattern. Will be updated once
  docs expose the canonical spec.

## [0.2.0] — 2026-06-06 (earlier same day)

### Added — Full video catalog (15 → 58 nodes)

**Wan family (16 nodes)**
- Wan 2.7: T2V, I2V, Video Edit, R2V
- Wan 2.6: T2V, I2V, V2V
- Wan 2.6 Flash: I2V, V2V (cheaper tier)
- Wan 2.5: T2V, I2V
- Wan 2.2 A14B Turbo: T2V, I2V, Speech-to-Video
- Wan Animate: Move, Replace

**Kling family completed (14 nodes)**
- Kling 3.0 single-shot + multi-shot (parses `<duration>:<prompt>` syntax)
- Motion control: 2.6 + 3.0
- Kling 2.6: T2V, I2V
- Kling 2.1: Standard, Pro, Master (T2V + I2V)
- Kling Avatar: Standard, Pro

**Bytedance family (8 nodes)**
- Seedance 2.0, Seedance 2.0 Fast (multi-modal)
- Seedance 1.5 Pro (integer duration, fixed_lens, generate_audio)
- V1 Pro: T2V, I2V, Fast I2V
- V1 Lite: T2V, I2V

**Hailuo family (5 nodes)**
- 02 Pro: T2V, I2V (5000-char prompt, prompt_optimizer)
- 02 Standard: T2V, I2V (1500-char prompt, optional end_image_url)
- 2.3 Standard I2V (10s+1080P forbidden per docs)

**Sora 2 family base (4 nodes)**
- sora-2 T2V/I2V; sora-2-pro T2V/I2V

**Utility nodes (4)**
- Topaz Video Upscale (1x/2x/4x)
- Infinitalk From Audio (lip-sync from portrait + audio)
- Grok Imagine Upscale, Grok Imagine Extend

### Changed
- `nodes/video/__init__.py` aggregates 9 sub-modules.
- Mutual-exclusivity rules enforced in every `build_input` (Wan 2.7 I2V
  sub-modes, Wan 2.7 R2V 5-ref cap, Hailuo 2.3 Std 10s+1080P, Seedance
  1.5 Pro 2-input cap, Sora 2 character cap).

## [0.1.0] — 2026-06-06 (initial release)

### Added — Initial release (15 nodes)
- `client/kie_client.py` v3 with dual-pattern dispatch (`run_market` + `run_veo`).
- `BaseKieMarketVideoNode`, `BaseKieMarketImageNode`, `BaseKieMarketAudioNode`,
  `BaseKieMarketTextNode`, `BaseKieVeoVideoNode` in `nodes/base.py`.
- Veo 3.1 (3 nodes): Quality, Fast, Lite.
- Kling 2.5 Turbo Pro: T2V, I2V.
- Bytedance Seedance 2.0 + Seedance 2.0 Fast (expanded in v0.2.0).
- Hailuo 2.3 Pro T2V + I2V (renamed in v0.2.0).
- HappyHorse 1.0: T2V, I2V (extended to 4 in v0.3.0).
- Grok Imagine: T2V, I2V.
- End-to-end smoke tests for Veo 3.1 and Grok (passed at ~$0.20 USD).

[0.3.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.3.0
[0.2.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.2.0
[0.1.0]: https://github.com/doomniel/genesis-kie-nodes/releases/tag/v0.1.0
