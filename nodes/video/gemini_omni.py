"""Google Gemini Omni multimodal video generation nodes (via GenesisLab proxy)."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..base import BaseKieMarketVideoNode, BaseKieNode, CATEGORY_VIDEO
from ...client import KieClient, KieError
from ...client.upload import upload_image_tensor, upload_video_frames

log = logging.getLogger("genesis_kie")


def _upload_batch_optional(image_tensor: Any) -> list[str]:
    if image_tensor is None:
        return []
    n = image_tensor.shape[0] if len(image_tensor.shape) >= 4 else 1
    return [upload_image_tensor(image_tensor[i:i + 1]) for i in range(n)]


def _csv(value: str) -> list[str]:
    if not value:
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


# ============================================================ Gemini Omni Video

class GeminiOmniVideo(BaseKieMarketVideoNode):
    """Gemini Omni Video — multimodal video generation.

    Slot budget per video (7 total):
    - Each image: 1 slot
    - Single video: 2 slots
    - Each character_id: 1 slot
    - audio_ids: max 1 ID, separate budget

    Note: audio_ids and character_ids are pre-created IDs (not URLs) from
    Gemini Omni Audio/Character creation nodes — they remain STRING inputs.
    """

    MODEL = "gemini-omni-video"
    POLL_INTERVAL_SECONDS = 5.0
    TIMEOUT_SECONDS = 1500.0

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Create a cinematic short film with a slow push-in camera.",
                }),
                "duration": (["4", "6", "8", "10"], {"default": "4"}),
            },
            "optional": {
                "reference_images": ("IMAGE", {"tooltip": "Reference image(s). Batch (1 slot each, max 7 combined)."}),
                "reference_video": ("IMAGE", {"tooltip": "Optional reference video as IMAGE batch (2 slots)."}),
                "fps": ("INT", {"default": 24, "min": 8, "max": 60, "step": 1}),
                "video_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 600.0, "step": 0.1}),
                "video_ends": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 600.0, "step": 0.1}),
                "audio_ids": ("STRING", {"default": "", "tooltip": "Pre-created audio_id (from Gemini Omni Audio node)."}),
                "character_ids": ("STRING", {"default": "", "tooltip": "Comma-separated character IDs (max 3)."}),
            },
        }

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "duration": str(kwargs["duration"]),
        }

        img_urls = _upload_batch_optional(kwargs.get("reference_images"))
        ref_video = kwargs.get("reference_video")
        chars = _csv((kwargs.get("character_ids") or "").strip())

        slots = len(img_urls) + (2 if ref_video is not None else 0) + len(chars)
        if slots > 7:
            raise ValueError(
                f"Gemini Omni: slot budget exceeded ({slots}/7). "
                "Each image=1, video=2, character=1."
            )
        if len(chars) > 3:
            raise ValueError(f"Gemini Omni: max 3 characters, got {len(chars)}.")

        if img_urls:
            body["image_urls"] = img_urls
        if ref_video is not None:
            fps = int(kwargs.get("fps", 24))
            video_url = upload_video_frames(ref_video, fps=fps)
            body["video_list"] = [{
                "url": video_url,
                "start": float(kwargs.get("video_start", 0.0)),
                "ends": float(kwargs.get("video_ends", 10.0)),
            }]
        if chars:
            body["character_id_list"] = chars

        audio_ids = _csv((kwargs.get("audio_ids") or "").strip())
        if audio_ids:
            if len(audio_ids) > 1:
                raise ValueError(f"Gemini Omni: max 1 audio_id, got {len(audio_ids)}.")
            body["audio_ids"] = audio_ids

        return body


# ============================================================ Gemini Omni Audio

class GeminiOmniAudioCreate(BaseKieNode):
    """Gemini Omni Audio — create a reusable voice. Returns kieAudioId STRING."""

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kie_audio_id",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_id": ("STRING", {"default": "achernar"}),
                "name": ("STRING", {"default": "achernar Narrator"}),
                "voice_description": ("STRING", {
                    "multiline": True,
                    "default": "A calm, clear, and friendly male voice suitable for tech explainers and daily conversation.",
                }),
            },
            "optional": {
                "example_dialogue": ("STRING", {"multiline": True, "default": "Hello, I am achernar."}),
            },
        }

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        with KieClient() as client:
            data = client.create_omni_audio(
                audio_id=kwargs["audio_id"],
                name=kwargs["name"],
                voice_description=kwargs["voice_description"],
                example_dialogue=(kwargs.get("example_dialogue") or "").strip(),
            )

        kie_audio_id = data.get("kieAudioId")
        if not kie_audio_id:
            raise KieError(f"Gemini Omni Audio create did not return kieAudioId. Response: {data}")
        log.info("Gemini Omni Audio created kieAudioId=%s", kie_audio_id)
        return (kie_audio_id,)


# ============================================================ Gemini Omni Character

class GeminiOmniCharacterCreate(BaseKieNode):
    """Gemini Omni Character — create a reusable character.

    Returns kieCharacterId STRING. Reference images via IMAGE batch.
    """

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("kie_character_id",)
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "character_id": ("STRING", {"default": "", "tooltip": "Source character template identifier."}),
                "name": ("STRING", {"default": "my_character_01"}),
                "character_description": ("STRING", {
                    "multiline": True,
                    "default": "A friendly cartoon character with expressive eyes and fluid movements.",
                }),
                "reference_images": ("IMAGE", {"tooltip": "Character reference image(s). Batch for multi-ref."}),
            },
        }

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        cid = (kwargs.get("character_id") or "").strip()
        if not cid:
            raise ValueError("Gemini Omni Character requires character_id.")
        urls = _upload_batch_optional(kwargs.get("reference_images"))
        if not urls:
            raise ValueError("Gemini Omni Character requires at least one reference image.")

        with KieClient() as client:
            data = client.create_omni_character(
                character_id=cid,
                name=kwargs["name"],
                character_description=kwargs["character_description"],
                image_urls=urls,
            )

        kie_char_id = (
            data.get("kieCharacterId")
            or data.get("character_id")
            or data.get("characterId")
        )
        if not kie_char_id:
            raise KieError(f"Gemini Omni Character create did not return a character ID. Response: {data}")
        log.info("Gemini Omni Character created id=%s", kie_char_id)
        return (kie_char_id,)


NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieGeminiOmniVideo": GeminiOmniVideo,
    "GenesisKieGeminiOmniAudioCreate": GeminiOmniAudioCreate,
    "GenesisKieGeminiOmniCharacterCreate": GeminiOmniCharacterCreate,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieGeminiOmniVideo": "Gemini Omni Video",
    "GenesisKieGeminiOmniAudioCreate": "Gemini Omni Audio (create voice)",
    "GenesisKieGeminiOmniCharacterCreate": "Gemini Omni Character (create)",
}
