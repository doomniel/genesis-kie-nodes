# Genesis-Kie

> ComfyUI custom nodes wrapping [Kie.ai](https://kie.ai)'s full multi-modal API surface — **150 nodes** across video, image, music, and LLM, abstracting **7 distinct endpoint patterns** behind a unified node interface.

[![Tag](https://img.shields.io/github/v/tag/doomniel/genesis-kie-nodes)](https://github.com/doomniel/genesis-kie-nodes/tags)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## ✨ At a glance

|  | Count | Examples |
|---|---:|---|
| 🎥 **Video** | 70 nodes | Wan 2.2 A14B, Kling 2.5, Sora 2, Bytedance Seedance, Hailuo, Veo 3.1, Runway Gen-4 Aleph |
| 🖼️ **Image** | 43 nodes | Flux 2.0, Seedream 4.5, Qwen-Image 2512, Ideogram 3, GPT 4o Image, Flux Kontext, Z-Image Turbo |
| 🎵 **Music / Audio** | 20 nodes | Suno V5 (11 ops), ElevenLabs TTS + Stem Separation, Lyrics, MIDI, WAV convert |
| 🤖 **LLM / Chat** | 17 nodes | GPT 5.5, Claude Opus 4.8, Gemini 3.1 Pro, GPT Codex (5 variants) |

**60+ underlying models** from OpenAI, Anthropic, Google, Meta, Black Forest Labs, ElevenLabs, Suno, Runway, ByteDance, Alibaba, Tencent — all behind a single API key.

---

## 🚀 Quickstart

### Requirements

- Python **3.10+**
- ComfyUI installation
- Kie.ai API key — get one at https://kie.ai

### Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/doomniel/genesis-kie-nodes.git
cd genesis-kie-nodes
pip install -r requirements.txt
```

### Configure

Create a `.env` file in `genesis-kie-nodes/`:

```ini
KIE_API_KEY=your-key-here
```

### Verify

Restart ComfyUI. Search for `GenesisKie/*` categories in the node browser. You should see Video, Image, Music, and LLM sub-categories with **150 nodes** total.

---

## 📚 Catalog by modality

### 🎥 Video (70 nodes, 12 families)

| Family | Nodes | Highlights |
|---|---:|---|
| Wan | 16 | Wan 2.2 A14B (T2V, I2V, image-ref multi), Wan 2.5 |
| Kling | 14 | Kling 2.5 Pro/Std (T2V, I2V, lipsync, image-ref) |
| Sora | 8 | Sora 2 Pro (until sunset Sep 2026), Sora 2 |
| Bytedance | 8 | Seedance, Seedance Lite, Seedance Pro |
| Hailuo | 5 | Hailuo 02, Hailuo 02 Pro (lipsync, I2V) |
| HappyHorse | 4 | First Frame, Last Frame, Multi-image, Lipsync |
| Grok | 4 | Grok video variants |
| Veo | 3 | Google Veo 3 / 3.1 Fast / 3.1 Pro (dedicated) |
| Runway | 3 | Gen-4 Turbo, Gen-4 Aleph (dedicated) |
| Gemini | 3 | Gemini video variants |
| Topaz | 1 | Topaz Video Upscale |
| Infinitalk | 1 | Infinitalk |

### 🖼️ Image (43 nodes, 11 families)

| Family | Nodes | Highlights |
|---|---:|---|
| Seedream | 7 | Seedream 4.5 T2I/I2I, edit, multi-image |
| Google | 7 | Imagen, Nano Banana variants |
| Ideogram | 6 | Ideogram 3 T2I, edit, magic prompt |
| Qwen | 5 | Qwen-Image 2512 (T2I, edit) |
| Flux-2 | 4 | Flux 2.0 dev, schnell, pro, ultra |
| GPT Image | 4 | GPT Image 2.0 + variants |
| Image Utils | 3 | Topaz Upscale, BG remove, restore |
| Wan Image | 2 | Wan 2.5 image variants |
| Grok Imagine | 2 | Grok image variants |
| GPT 4o (dedicated) | 1 | GPT 4o Image (own endpoint) |
| Flux Kontext (dedicated) | 1 | BFL Flux Kontext Pro/Max |
| Z-image | 1 | Z-Image Turbo |

### 🎵 Music / Audio (20 nodes, 2 families)

| Family | Nodes | Endpoints |
|---|---:|---|
| **ElevenLabs (Market)** | 4 | audio-isolation, TTS-multilingual-v2, TTS-turbo-2.5, text-to-dialogue-v3 |
| **Suno Music Gen (Dedicated)** | 11 | Generate, Extend, UploadCover, UploadExtend, MusicCover, AddInstrumental, AddVocals, BoostStyle, ReplaceSection, Persona, Mashup |
| **Suno Utilities (Dedicated)** | 5 | GenerateLyrics, TimestampedLyrics, ConvertToWAV, VocalRemoval, GenerateMIDI |

### 🤖 LLM / Chat (17 nodes, 4 families)

| Family | Nodes | Pattern |
|---|---:|---|
| GPT | 3 | GPT 5.2 (chat), GPT 5.4 (responses), GPT 5.5 (responses) |
| Claude | 7 | Opus 4.5/4.6/4.7/4.8, Sonnet 4.5/4.6, Haiku 4.5 |
| Codex | 1 | GPT Codex (5 versions via dropdown: 5/5.1/5.2/5.3/5.4-codex) |
| Gemini | 6 | 2.5 Pro, 2.5 Flash, 3 Pro, 3 Flash, 3.1 Pro, 3.5 Flash |

---

## 🔌 Endpoint patterns

Kie.ai exposes **7 distinct API patterns**. This package abstracts all of them behind a single `KieClient` with consistent error handling:

| # | Pattern | Used by | Sync? | Result extraction |
|---|---|---|---|---|
| 1 | Market generic | Seedance, Kling, Hailuo, ElevenLabs, ... | Async + polling | `_parsed_result.resultUrls` |
| 2 | Veo dedicated | Google Veo | Async + polling | `response.resultUrls` (successFlag-based) |
| 3 | Runway dedicated | Runway Gen-4 | Async + polling | `response.resultUrls` (status-based) |
| 4 | 4o Image dedicated | GPT 4o Image | Async + polling | `response.resultUrls` |
| 5 | Flux Kontext dedicated | BFL Flux Kontext | Async + polling | `response.resultUrls` / `resultImageUrl` |
| 6 | Suno dedicated | Suno + utilities (lyrics, wav, midi, vocal) | Async + polling | `response.sunoData[]` (status enum) |
| 7 | Chat / LLM | GPT, Claude, Codex, Gemini | **Synchronous** | varies by sub-pattern (3 shapes) |

The 7th pattern itself has 3 sub-shapes:
- **OpenAI Chat Completions** (GPT 5.2 + all Gemini): `choices[0].message.content`
- **OpenAI Responses API** (GPT 5.4/5.5 + Codex): `output[].content[].text` + reasoning effort dial
- **Anthropic Messages** (all Claude): `content[].text` + optional extended thinking

---

## 📤 Output shapes

| Modality | RETURN_TYPES | Why this shape |
|---|---|---|
| Video | `(IMAGE, INT, INT)` frames + fps + frame_count | Direct ComfyUI VideoPlayer compatibility |
| Image | `(IMAGE,)` tensor B×H×W×C | Direct Save/Preview node input |
| Audio (ElevenLabs) | `(STRING,)` audio_path | Single file, simple downstream |
| Suno Music | `(STRING, STRING, STRING)` audio_path + audio_id + all_paths_csv | `audio_id` enables Extend/Cover/AddVocals chaining |
| Suno Lyrics | `(STRING,)` text | Concatenated 2-3 variations |
| Vocal Removal | `(STRING, STRING, STRING)` vocals + instrumental + all_stems | Multi-stem output |
| Suno WAV / MIDI | `(STRING,)` file_path | Single converted file |
| LLM / Chat | `(STRING, INT)` text + tokens_used | Cost tracking built-in |

---

## 🧪 Smoke testing

Each phase ships its own smoke runner. From repo root:

```bash
# Quick patterns validation (~$0.015)
python smoke_phase5.py all_cheap   # gemini_flash + claude_haiku + gpt 5.2

# Music validation (~$0.05)
python smoke_phase4.py lyrics       # Suno lyrics
python smoke_phase4.py el_tts       # ElevenLabs TTS
python smoke_phase4.py suno         # Suno Generate Music

# Image validation (~$0.10)
python smoke_phase3.py kontext_pro  # Flux Kontext
python smoke_phase3.py gpt4o_image  # GPT 4o Image
python smoke_phase3.py seedream     # Seedream 4.5
```

---

## 💰 Cost gotchas

Approximate baselines (rates vary, check kie.ai pricing):

| Family | Baseline | Notes |
|---|---:|---|
| Suno Lyrics | $0.01 | Returns 2-3 lyric variations |
| Gemini 3.5 Flash | $0.003 | Cheapest LLM in the catalog |
| Claude Haiku 4.5 | $0.005 | Fast tier |
| GPT 5.2 | $0.005 | Cheap GPT baseline |
| ElevenLabs TTS | $0.02 | Per ~200 chars |
| Suno Generate Music | $0.04 | Returns 2 audio variants |
| Gemini 3.1 Pro | $0.02 | Frontier reasoning |
| GPT 5.5 (reasoning) | $0.04 | Responses API |
| Claude Opus 4.8 | $0.05 | Frontier coding/writing |
| **GPT Codex** | **$0.37 baseline ⚠️** | Codex injects ~2500-token system prompt regardless of your input |
| Suno Persona / WAV / MIDI | varies | Often free post-generation |
| Veo / Runway video | $0.50-$2.00 | Premium tier |

For real-time balance checks before workflow run, see the planned `KieCreditsCheck` node in [Roadmap](#roadmap).

---

## 🗺️ Roadmap

- [ ] **v0.7.0** — Music Video Generation + Sounds Generation + Gemini Omni Character (~5-8 nodes)
- [ ] **v0.8.0** — Example `.json` workflows demonstrating common pipelines
- [ ] **v0.9.0** — `KieCreditsCheck` node (balance preview before workflow execution)
- [ ] **v1.0.0** — Production deploy guide + GenesisLab multi-tenant integration

---

## 📋 Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full version history.

Recent releases:
- **v0.6.0** — 17 LLM nodes (GPT, Claude, Codex, Gemini) + 7th endpoint pattern
- **v0.5.0** — 20 music nodes (ElevenLabs + Suno) + 6th endpoint pattern
- **v0.4.0** — 43 image nodes across 11 families
- **v0.3.0** — 70 video nodes, full Kie Market coverage

---

## 🤝 Contributing

PRs welcome. When adding new Kie.ai endpoints:

1. Add the API spec to the relevant `nodes/<modality>/` file
2. Reuse one of the existing base classes (`BaseKieMarketVideoNode`, `BaseKieChatNode`, etc) or extend `BaseKieNode` if it's a new pattern
3. Smoke-test against a real Kie.ai call before opening the PR
4. Update the catalog table in `README.md`

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

## 👤 Author

Built by [@doomniel](https://github.com/doomniel) as the API layer for [GenesisLab](https://genesislab.app), a multi-tenant ComfyUI SaaS platform.
