# genesis-kie-nodes

**ComfyUI custom nodes for [Kie.ai](https://kie.ai) — 70 video models, one API key.**

Production-grade integration of Kie.ai into ComfyUI. Every node maps 1:1 to a real Kie endpoint with parameters extracted verbatim from the upstream OpenAPI specs — **no invented parameters, no fictional defaults**.

Built and maintained by [@doomniel](https://github.com/doomniel) as the unified video provider for [GenesisLab](https://genesislab.top).

---

## Why this exists

If you've used Kie.ai before, you know the Market is a goldmine: Veo 3.1, Sora 2, Wan 2.7, Kling 3.0, HappyHorse 1.0, Bytedance Seedance, Hailuo, Runway Gen-4, Gemini Omni, Topaz, Infinitalk — all behind a single API key, with prices typically 30–60% below the official providers. But integrating it into ComfyUI requires:

- Knowing which models exist (the docs sprawl across ~80 endpoints).
- Knowing each endpoint's quirks (Wan 2.6 uses `image_urls: [url]`, Wan 2.5 uses `image_url: url`, Sora 2 Pro requires `size: standard/high`, Runway uses camelCase without an `input` wrapper, Gemini Omni Audio returns `code: 0` for success…).
- Knowing the mutual-exclusivity rules (Wan 2.7 I2V has 3 sub-modes that can't be combined, Wan 2.7 R2V caps refs at 5, Hailuo 2.3 Std forbids 10s @ 1080P, Runway 1080p can't be extended, Gemini Omni's 7-slot budget…).
- Handling THREE distinct task lifecycles (Market `createTask`+`recordInfo`, Veo `veo/generate`+`record-info`, Runway `runway/generate`+`record-detail`).

This package handles all of that.

---

## Catalog (70 nodes)

### Video (70)

| Family | Count | Models |
|---|---|---|
| **Wan** | 16 | 2.7 T2V/I2V/Video Edit/R2V · 2.6 T2V/I2V/V2V · 2.6 Flash I2V/V2V · 2.5 T2V/I2V · 2.2 A14B Turbo T2V/I2V/Speech · Animate Move/Replace |
| **Kling** | 14 | 3.0 single + multi-shot · motion-control 2.6 + 3.0 · 2.6 T2V/I2V · 2.5 Turbo Pro T2V/I2V · 2.1 Std/Pro/Master · Avatar Std/Pro |
| **Sora 2** | 8 | sora-2 T2V/I2V · sora-2-pro T2V/I2V (size: standard/high) · Watermark Remover · Characters · Characters Pro · Pro Storyboard |
| **Bytedance** | 8 | Seedance 2.0 · 2.0 Fast · 1.5 Pro · V1 Pro T2V/I2V/Fast I2V · V1 Lite T2V/I2V |
| **Hailuo** | 5 | 02 Pro T2V/I2V · 02 Standard T2V/I2V · 2.3 Standard I2V |
| **Grok Imagine** | 4 | T2V · I2V · Upscale · Extend |
| **HappyHorse 1.0** | 4 | T2V · I2V · Reference-to-Video (1-9 refs) · Video Edit (V2V) |
| **Veo 3.1** | 3 | Quality · Fast · Lite |
| **Runway** | 3 | Gen-4 Turbo · Extend · Aleph (V2V) |
| **Gemini Omni** | 3 | Omni Video (multimodal) · Audio create · Character create |
| **Topaz** | 1 | Video Upscale (1x/2x/4x) |
| **Infinitalk** | 1 | Audio → lip-synced video |

Coming next (Phase 3): Image models (~30 nodes — Seedream, Imagen 4, Flux 2, Nano Banana, GPT Image, Topaz Image, Recraft, Ideogram, Qwen, 4o, Flux Kontext, Wan 2.7 Image, Z-Image, Grok Imagine Image).

---

## Requirements

- ComfyUI (any recent version)
- Python 3.10+
- A Kie.ai API key — get one at <https://kie.ai/api-key>

---

## Installation

### Method 1: ComfyUI Manager (recommended once published)

Search `genesis-kie-nodes` in ComfyUI Manager and install.

### Method 2: Git clone

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/doomniel/genesis-kie-nodes.git
cd genesis-kie-nodes
pip install -r requirements.txt
```

### Configuration

Set your API key via one of:

**A. Environment variable** (recommended):

```bash
export KIE_API_KEY="sk-..."
```

**B. `.env` file** in the package root:

```ini
KIE_API_KEY=sk-...
```

The package auto-loads `.env` on import if present.

Restart ComfyUI and look for `Kie — *` nodes in the node browser.

---

## Quick start

Most nodes follow the same pattern:

1. Drop the node onto the canvas.
2. Fill in the prompt and any required URLs.
3. Connect the `VIDEO` output to a Save Video node (or chain it into further processing).

The node will:

1. Call the appropriate Kie endpoint (Market `createTask`, Veo `veo/generate`, or Runway `runway/generate`).
2. Poll the matching detail endpoint every few seconds.
3. Download the result MP4 to ComfyUI's `output/` directory.
4. Emit the local file path on its `VIDEO` output.

### Example: Wan 2.7 text-to-video

```
[Kie — Wan 2.7 (T2V)]
  prompt: "A drone shot over neon Tokyo at night, slow push-in."
  resolution: 1080p
  ratio: 16:9
  duration: 5
        ↓
[Save Video]
```

### Example: Runway Gen-4 Turbo + Extend (chained)

```
[Kie — Runway Gen-4 Turbo]                    [Kie — Runway Extend]
  prompt: "A cat dances in a disco room"   →    taskId: <from previous>
  quality: 720p                            →    prompt: "The cat speeds up"
  duration: 5                              →    quality: 720p
```

(Runway 1080p videos can't be extended — the node enforces this.)

### Example: Gemini Omni with reusable voice + character

```
[Kie — Gemini Omni Audio (create voice)]
  audio_id: "achernar"
  name: "narrator_v1"
  description: "Calm male voice, tech explainers"
        ↓ kie_audio_id (string)
        
[Kie — Gemini Omni Character (create)]
  character_id: "<template>"
  image_urls: "https://.../a.png, https://.../b.png"
        ↓ kie_character_id (string)
        
[Kie — Gemini Omni Video]
  prompt: "A cinematic close-up..."
  audio_ids: <connected from Audio node>
  character_ids: <connected from Character node>
  duration: 8
```

The slot budget (7 total: image=1, video=2, character=1) is validated **before** spending credits.

---

## Architecture

```
genesis-kie-nodes/
├── __init__.py                    # Registers all node mappings
├── client/
│   ├── kie_client.py              # HTTPx-based client (Market + Veo + Runway + Omni)
│   ├── auth.py                    # Bearer-token auth
│   ├── exceptions.py              # Typed errors
│   └── download.py                # Polling + asset download
├── nodes/
│   ├── base.py                    # BaseKieMarketVideoNode + base classes
│   ├── video/
│   │   ├── veo31.py               # Veo 3.1 (3 nodes — /veo3-api/)
│   │   ├── kling.py               # Kling family (14 nodes)
│   │   ├── wan.py                 # Wan family (16 nodes)
│   │   ├── seedance.py            # Bytedance family (8 nodes)
│   │   ├── hailuo.py              # Hailuo family (5 nodes)
│   │   ├── sora.py                # Sora 2 family (8 nodes)
│   │   ├── happyhorse.py          # HappyHorse 1.0 (4 nodes)
│   │   ├── grok.py                # Grok Imagine T2V/I2V (2 nodes)
│   │   ├── runway.py              # Runway Gen-4 + Aleph (3 nodes — dedicated API)
│   │   ├── gemini_omni.py         # Gemini Omni multimodal (3 nodes — sync helpers)
│   │   └── utils.py               # Topaz + Infinitalk + Grok Upscale/Extend
│   ├── image/                     # (Phase 3 — coming)
│   ├── music/                     # (Phase 4 — coming)
│   └── llm/                       # (Phase 5 — coming)
└── tests_local/                   # End-to-end smoke tests
```

### Three endpoint patterns under one client

`KieClient` handles three distinct Kie endpoint patterns transparently:

1. **Market generic** (`run_market`) — covers ~80% of nodes. `POST /api/v1/jobs/createTask` with `{model, input: {...}}`, poll `GET /api/v1/jobs/recordInfo`.

2. **Veo dedicated** (`run_veo`) — for Veo 3.1. `POST /api/v1/veo/generate` with parameters at the top level (no `input` wrapper), poll `GET /api/v1/veo/record-info`. Status lives in `successFlag` (0/1/2/3) not in `state`.

3. **Runway dedicated** (`run_runway`) — for Runway Gen-4 + Aleph. Uses `POST /api/v1/runway/generate` (or `/extend`, `/aleph/generate`) with camelCase parameters at the top level. Polls `GET /api/v1/runway/record-detail`. Result lives in `data.videoInfo.videoUrl` (not `resultJson`).

Plus two synchronous helper endpoints for Gemini Omni:

- `POST /api/v1/omni/audio/create` — voice creation, returns `kieAudioId` immediately.
- `POST /api/v1/omni/character/create` — character creation, returns `kieCharacterId` immediately. *(Note: body schema inferred from audio endpoint pattern; will be confirmed against canonical spec when docs.kie.ai exposes it.)*

### `BaseKieMarketVideoNode`

All Market-pattern video nodes inherit from `BaseKieMarketVideoNode`, which provides:

- `MODEL: ClassVar[str]` — the Kie endpoint slug
- `POLL_INTERVAL_SECONDS` / `TIMEOUT_SECONDS` — per-model defaults
- `RETURN_TYPES = ("VIDEO",)`
- `FUNCTION = "execute"`
- `execute()` — the universal lifecycle (build → submit → poll → download → return)

Subclasses only implement:

- `INPUT_TYPES()` — UI definition with the node's specific parameters
- `build_input(**kwargs) -> dict` — translates UI inputs into the Kie API body, **with input validation**

Runway nodes use `_BaseRunwayNode` instead (similar shape but builds camelCase bodies and uses `KieClient.run_runway`).

---

## A note on parameter fidelity

Every parameter, enum, default, and constraint in this package came from the **`.md` versions** of `docs.kie.ai` endpoints, which expose the raw OpenAPI specs.

Where the OpenAPI omits or contradicts itself (e.g. Sora 2 Pro Storyboard's body schema, Gemini Omni Character's request path), the cURL examples in the docs were used as ground truth and clearly marked as **inferred** in the module docstring. The exact source for each family is documented in that file.

If you spot a parameter that doesn't behave as documented in this repo: please open an issue with the exact docs URL and the upstream cURL example. **I will not add parameters that aren't in the upstream specs.**

---

## Roadmap

- [x] **v0.1.0** — Batch 1: Veo 3.1 + first nodes (15 total)
- [x] **v0.2.0** — Batch 2: full video catalog minus Runway/Omni (58 total)
- [x] **v0.3.0** — Batch 3: complete video catalog (70 total) ← **you are here**
- [ ] **v0.4.0** — Image models (~30 nodes: Seedream, Imagen 4, Flux 2, Nano Banana, GPT Image, Topaz Image, Recraft, Ideogram, Qwen, 4o, Flux Kontext, Wan 2.7 Image, Z-Image, Grok Imagine Image)
- [ ] **v0.5.0** — Music models (Suno API + ElevenLabs)
- [ ] **v0.6.0** — Chat models (Claude, GPT, Gemini, Codex)
- [ ] **v1.0.0** — GenesisLab integration (kie.genesislab.top, multi-tenant billing)

---

## Pricing reality check

Kie.ai consolidates these providers. As of June 2026, the headline savings vs official APIs:

| Model | Official | via Kie | Savings |
|---|---|---|---|
| Sora 2 std (10s, 720p) | $1.00 | $0.15 | ~85% |
| Sora 2 Pro HD (10s, 1080p) | $2.00 | $1.00 | ~50% |
| Runway Gen-4 Turbo (10s, 720p) | $0.50 | varies | varies |
| Veo 3.1 Quality (8s, 1080p) | varies | competitive | varies |
| HappyHorse 1.0 (5s, 720p) | $0.625 | varies | varies |
| Kling 3.0 (5s, 1080p) | $1.40 | ~$0.55 | ~60% |

Prices fluctuate. Use **`Common API → Get Account Credits`** to check live cost-per-call before launching production runs.

---

## License

MIT — see `LICENSE`.

---

## Acknowledgments

- [Kie.ai](https://kie.ai) for aggregating an absurd number of frontier video models behind one API.
- The ComfyUI community for the node interface standard.
- Anthropic Claude for pair-coding the bulk of this catalog (literally — go look at the commit messages).

---

## Author

**[@doomniel](https://github.com/doomniel)** — Damniel, transmedia designer & creative technologist building [GenesisLab](https://genesislab.top) (multi-tenant ComfyUI SaaS on Proxmox). Based in CDMX, available for collaborations: [Mojo Supermarket](https://mojosupermarket.com), Havas, etc.
