# v0.4.0 — Image catalog (113 nodes total)

Adds **43 image nodes across 11 families**, plus 2 new endpoint patterns. Total catalog grows to **113 nodes**.

## 🖼️ Coverage (43 new nodes)

| Family | Nodes | Highlights |
|---|---:|---|
| Seedream | 7 | Seedream 4.5 T2I/I2I, edit, multi-image |
| Google | 7 | Imagen, Nano Banana variants |
| Ideogram | 6 | Ideogram 3 T2I, edit, magic prompt |
| Qwen | 5 | Qwen-Image 2512 (T2I, edit) |
| Flux-2 | 4 | Flux 2.0 dev, schnell, pro, ultra |
| GPT Image | 4 | GPT Image 2.0 variants |
| Image Utils | 3 | Topaz Upscale, BG remove, restore |
| Wan Image | 2 | Wan 2.5 image variants |
| Grok Imagine | 2 | |
| **GPT 4o (dedicated)** | 1 | New endpoint pattern (4th) |
| **Flux Kontext (dedicated)** | 1 | New endpoint pattern (5th) |
| Z-image | 1 | Z-Image Turbo |

## 🔌 New endpoint patterns

- **4th: 4o Image dedicated** — `/api/v1/gpt4o-image/generate` (Veo-style polling)
- **5th: Flux Kontext dedicated** — `/api/v1/flux/kontext/*`

`KieClient` now exposes 5 endpoint patterns total: Market + Veo + Runway + 4o Image + Flux Kontext.

## 🔧 Fixed

- Flux Kontext response parsing handles both `resultUrls` (array) and `resultImageUrl` (singular) shapes.

## 📤 Output

`(IMAGE,)` tensor B×H×W×C for direct Save/Preview node compatibility.

## 🧪 Smoke validated

3/3 PASS:
- GPT 4o Image — 42s, 1×1254×1254×3
- Flux Kontext Pro — 20s, 1×752×1392×3
- Seedream 4.5 T2I — 18s, 1×2048×2048×3

**Total: 113 nodes (70 video + 43 image).**
