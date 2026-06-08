# v0.3.0 — Video catalog complete (70 nodes)

First public release. Closes the video catalog with **70 nodes across 12 families**, covering Kie.ai's full video Market + dedicated endpoints.

## 🎥 Coverage

| Family | Nodes | Highlights |
|---|---:|---|
| Wan | 16 | Wan 2.2 A14B (T2V, I2V, image-ref), Wan 2.5 |
| Kling | 14 | Kling 2.5 Pro/Standard, lipsync, image-ref |
| Sora | 8 | Sora 2, Sora 2 Pro (note: API sunset Sep 2026) |
| Bytedance | 8 | Seedance, Seedance Lite, Seedance Pro |
| Hailuo | 5 | Hailuo 02, Hailuo 02 Pro |
| HappyHorse | 4 | First/Last Frame, Multi-image, Lipsync |
| Grok | 4 | Grok video variants |
| Veo (dedicated) | 3 | Veo 3, 3.1 Fast, 3.1 Pro |
| Runway (dedicated) | 3 | Gen-4 Turbo, Gen-4 Aleph |
| Gemini | 3 | video variants |
| Topaz | 1 | Video Upscale |
| Infinitalk | 1 | |

## 🔌 3 endpoint patterns

- **Market generic** — `/api/v1/jobs/createTask` for most families
- **Veo dedicated** — `successFlag`-based polling
- **Runway dedicated** — status-string polling

## 📤 Output

`(IMAGE, INT, INT)` — frames tensor B×H×W×C + fps + frame_count. Direct ComfyUI VideoPlayer compatibility.

## 📦 Install

```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/doomniel/genesis-kie-nodes.git
cd genesis-kie-nodes
pip install -r requirements.txt
echo "KIE_API_KEY=your-key" > .env
```

**Total: 70 nodes.**
