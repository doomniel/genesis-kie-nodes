# ComfyUI Genesis Kie

Native ComfyUI custom nodes for [Kie.ai](https://kie.ai) — unified multi-modal AI generation API offering 30-95% lower prices than fal.ai for the same models.

## Features

- **Video**: Veo 3.1, Kling 3.0, Seedance 2.0, Hailuo 2.3, Wan 2.5/2.6/2.7, HappyHorse, Grok Imagine, Runway, Sora 2
- **Image**: GPT Image 2, Nano Banana Pro, FLUX.2, Seedream 4.5/5.0, Ideogram V3, Imagen 4, Qwen, Z-Image
- **Music**: Suno V5/V4.5, ElevenLabs (TTS, dialogue, isolation)
- **LLMs**: Claude (Opus, Sonnet, Haiku), GPT (5.x, Codex), Gemini (Pro, Flash)

## Why this exists

Kie.ai is consistently 20-95% cheaper than fal.ai for the same closed-source models:

- Veo 3.1 Extend Lite: 95% off
- GPT Image 2: 86% off
- MeiGen InfiniteTalk: 85% off
- Grok Imagine Extend: 81% off
- Claude/GPT/Gemini LLMs: ~72% off

This package provides native ComfyUI nodes that talk directly to Kie.ai — no proxy translation, no parsing fal.ai responses, just clean async task creation and polling.

## Installation

### ComfyUI Manager (when published)

Coming soon.

### Manual installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/doomniel/genesis-kie-nodes.git
cd genesis-kie-nodes
pip install -r requirements.txt
```

Restart ComfyUI.

## Configuration

Set your Kie.ai API key as an environment variable before launching ComfyUI:

```bash
# Linux/Mac
export KIE_API_KEY="your_key_here"

# Windows PowerShell
$env:KIE_API_KEY = "your_key_here"
```

Or alternatively, place it in a `.env` file in the ComfyUI root directory.

## Architecture

```
genesis-kie-nodes/
├── client/                  # HTTP client + auth + download helpers
│   ├── kie_client.py        # POST createTask + polling
│   ├── auth.py              # API key management
│   ├── exceptions.py        # KieError, KieTimeoutError, KieAuthError
│   └── download.py          # Download result URLs as ComfyUI tensors
├── nodes/
│   ├── base.py              # BaseKieNode + BaseKieVideoNode + BaseKieImageNode
│   ├── video/               # Veo, Kling, Seedance, Hailuo, Wan, etc.
│   ├── image/               # GPT Image, Nano Banana, FLUX, etc.
│   ├── music/               # Suno, ElevenLabs
│   └── llm/                 # Claude, GPT, Gemini
└── __init__.py              # Node registration
```

## API model

Kie.ai uses an async task-based model:

```python
# 1. Create task
POST https://api.kie.ai/api/v1/jobs/createTask
body: { model: "google/veo-3.1-fast", input: {...} }
→ { taskId: "task_..." }

# 2. Poll task status
GET https://api.kie.ai/api/v1/jobs/{taskId}
→ { status: "completed", output: { video_url: "..." } }
```

All polling logic lives in `client/kie_client.py`. Individual nodes just declare their model name and input parameters.

## License

MIT — see [LICENSE](LICENSE).

## Disclaimer

Not affiliated with Kie.ai. Built by [doomniel](https://github.com/doomniel) for use in [GenesisLab](https://genesislab.top) and the wider ComfyUI community.
