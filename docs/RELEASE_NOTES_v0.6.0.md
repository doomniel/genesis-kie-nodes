# v0.6.0 — LLM / Chat catalog (150 nodes total) 🎉

Adds **17 LLM / Chat nodes** covering GPT, Claude, Codex, and Gemini. Introduces the 7th and final core endpoint pattern: synchronous chat completions. Total catalog reaches **150 nodes**.

## 🤖 Coverage (17 new nodes)

### GPT (3)
- GPT 5.2 — daily-driver chat (Chat Completions API)
- GPT 5.4 — reasoning model (Responses API + `reasoning_effort` dial)
- GPT 5.5 — current flagship (Responses API + reasoning)

### Claude (7)
- Opus 4.5 / 4.6 / 4.7 / 4.8 (frontier, leads Intelligence Index at 61.4)
- Sonnet 4.5 / 4.6 (best for writing + instruction following)
- Haiku 4.5 (fast tier)

All Claude nodes support optional **extended thinking** with `thinking` + `thinking_budget` controls.

### Codex (1)
- GPT Codex with version dropdown:
  - `gpt-5-codex`, `gpt-5.1-codex`, `gpt-5.2-codex`, `gpt-5.3-codex`, `gpt-5.4-codex`
- Defaults to `gpt-5.4-codex` (flagship).

### Gemini (6)
- 2.5 Pro / 2.5 Flash
- 3 Pro / 3 Flash
- 3.1 Pro (frontier, leads GPQA Diamond 94.3%)
- 3.5 Flash (best price-to-performance at frontier)

## 🔌 7th endpoint pattern: Chat (SYNCHRONOUS)

The first synchronous endpoint pattern in the package. No taskId, no polling, single POST returns the model's response in 1-30s.

Three sub-shapes share the same `KieClient.chat_completion()` infrastructure:

| Sub-pattern | Used by | Body shape | Response shape |
|---|---|---|---|
| OpenAI Chat Completions | GPT 5.2 + all Gemini | `{messages, max_tokens, temperature, stream}` | `choices[0].message.content` |
| OpenAI Responses API | GPT 5.4/5.5 + Codex | `{model, input, reasoning, stream}` | `output[].content[].text` |
| Anthropic Messages | all Claude | `{model, messages, system, thinking}` | `content[].text` |

`KieClient` now exposes **7 endpoint patterns** total — full coverage of Kie.ai's public API surface.

## 📤 Outputs

All 17 LLM nodes share the same shape:
- `text` (STRING) — the assistant's reply
- `tokens_used` (INT) — total tokens consumed, for cost tracking

## 🖼️ Multimodal support

All 17 nodes accept an optional `image_url` input. Models that support vision will process it; text-only models will return an error from Kie.ai.

## 🔧 Fixed (during validation)

- **Gemini "model not supported" 422**: `_GeminiChatBase` now strips `model` from the request body. Kie's Gemini gateway rejects bodies containing it because the endpoint URL already identifies the model. GPT and Claude bases still include `model` as expected.
- **Codex `max_output_tokens` 500 bug**: removed `max_output_tokens` from Responses API body. Kie.ai's `/api/v1/responses` returns `{"code":500,"msg":"Server exception"}` when this field is present. The field is non-standard in the Responses API spec anyway — the model uses its own default limit. Verified via 3-variant debug script.

## 💰 Cost gotcha — Codex baseline

GPT Codex injects a **~2500-token system prompt** baseline regardless of your user input length. Minimum cost per call is **~$0.37 USD** (verified empirically). Plan accordingly for high-volume use; for cheaper coding queries, GPT 5.2 or Claude Sonnet 4.6 are 50-70× more economical.

## 🧪 Smoke validated

All 3 chat sub-patterns confirmed working end-to-end:
- ✅ OpenAI Chat Completions — GPT 5.2 (13.2s), Gemini 3.1 Pro (7.8s)
- ✅ Anthropic Messages — Claude Haiku 4.5 (5.7s), Claude Opus 4.8 (4.3s)
- ✅ OpenAI Responses API — GPT 5.5 reasoning=low (5.3s), Codex 5.4 reasoning=medium

## 📊 Total catalog

**150 nodes** across 4 modalities and 7 endpoint patterns:
- 🎥 70 video
- 🖼️ 43 image
- 🎵 20 music
- 🤖 17 LLM

This release closes the **1.0-equivalent core coverage** of Kie.ai's API surface.
