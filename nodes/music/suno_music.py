"""Suno music generation nodes (via Kie.ai's DEDICATED endpoints).

Covers 11 Suno music-generation endpoints:

  Core generation:
  - Generate Music                  POST /api/v1/generate
  - Extend Music                    POST /api/v1/generate/extend

  Cover (transform existing tracks):
  - Upload And Cover Audio          POST /api/v1/generate/upload-cover
  - Upload And Extend Audio         POST /api/v1/generate/upload-extend
  - Generate Music Cover            POST /api/v1/generate/cover-suno

  Layering:
  - Add Instrumental to Music       POST /api/v1/generate/add-instrumental
  - Add Vocals to Music             POST /api/v1/generate/add-vocals

  Editing:
  - Boost Music Style               POST /api/v1/generate/boost-style
  - Replace Music Section           POST /api/v1/generate/replace-section

  Creative:
  - Generate Persona                POST /api/v1/generate/generate-persona
  - Generate Mashup Music           POST /api/v1/generate/mashup

All endpoints share the polling URL /api/v1/generate/record-info.

Schemas verbatim from docs.kie.ai cURL examples. Each node returns
(audio_path, audio_id, all_paths_csv) — the audio_id is what you connect
into downstream Extend/AddVocals/Cover nodes to chain operations.
"""

from __future__ import annotations

from typing import Any, ClassVar

from ..base import BaseKieSunoMusicNode


# Suno model versions (latest first, per docs.kie.ai pricing page).
_SUNO_MODELS = ["V5", "V4_5PLUS", "V4_5", "V4", "V3_5"]
_VOCAL_GENDERS = ["", "m", "f"]


# ============================================================ CORE generation

class SunoGenerateMusic(BaseKieSunoMusicNode):
    """Generate music from text prompt (Suno's core endpoint).

    Two modes:
    - customMode=false: only prompt required (quick start).
    - customMode=true: full control over style, title, vocalGender, weights.

    Set instrumental=true for music without vocals.
    """

    MODEL = "suno-generate"
    CREATE_ENDPOINT = "/api/v1/generate"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A calm and relaxing piano track with soft melodies",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
                "custom_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "false = quick start (prompt only). true = full custom control.",
                }),
                "instrumental": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "true = no vocals. false = with vocals.",
                }),
            },
            "optional": {
                "style": ("STRING", {
                    "default": "",
                    "tooltip": "(custom mode) Music style/genre (e.g., 'Classical', 'Hip-hop').",
                }),
                "title": ("STRING", {
                    "default": "",
                    "tooltip": "(custom mode) Track title.",
                }),
                "negative_tags": ("STRING", {
                    "default": "",
                    "tooltip": "Styles to EXCLUDE (comma-separated).",
                }),
                "vocal_gender": (_VOCAL_GENDERS, {
                    "default": "",
                    "tooltip": "m/f for male/female vocals, empty = automatic.",
                }),
                "style_weight": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "weirdness_constraint": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "audio_weight": ("FLOAT", {
                    "default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05,
                }),
                "persona_id": ("STRING", {
                    "default": "",
                    "tooltip": "Optional Suno persona ID for voice consistency.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        body: dict[str, Any] = {
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "customMode": bool(kwargs["custom_mode"]),
            "instrumental": bool(kwargs["instrumental"]),
        }
        for src, dst in [
            ("style", "style"),
            ("title", "title"),
            ("negative_tags", "negativeTags"),
            ("persona_id", "personaId"),
        ]:
            value = (kwargs.get(src) or "").strip()
            if value:
                body[dst] = value

        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg

        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


class SunoExtendMusic(BaseKieSunoMusicNode):
    """Extend an existing Suno-generated track.

    Takes the ``audio_id`` from a previous Generate/Cover task and adds
    more music starting at ``continue_at`` seconds.
    """

    MODEL = "suno-extend"
    CREATE_ENDPOINT = "/api/v1/generate/extend"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id from a previous Suno task.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Extend with a gentle bridge and outro.",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
                "continue_at": ("INT", {
                    "default": 60, "min": 0, "max": 600,
                    "tooltip": "Continue from this many seconds into the source.",
                }),
            },
            "optional": {
                "default_param_flag": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reuse the source track's parameters.",
                }),
                "style": ("STRING", {"default": ""}),
                "title": ("STRING", {"default": ""}),
                "negative_tags": ("STRING", {"default": ""}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": ""}),
                "style_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weirdness_constraint": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "persona_id": ("STRING", {"default": ""}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not audio_id:
            raise ValueError("Suno Extend requires audio_id.")

        body: dict[str, Any] = {
            "audioId": audio_id,
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "continueAt": int(kwargs["continue_at"]),
            "defaultParamFlag": bool(kwargs.get("default_param_flag", True)),
        }
        for src, dst in [
            ("style", "style"), ("title", "title"),
            ("negative_tags", "negativeTags"), ("persona_id", "personaId"),
        ]:
            v = (kwargs.get(src) or "").strip()
            if v:
                body[dst] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


# ============================================================ COVER family

class SunoUploadAndCover(BaseKieSunoMusicNode):
    """Upload an audio URL and create a stylistic cover of it.

    Use this to transform an existing track (your own upload, or any
    publicly accessible audio URL) into a new style while preserving
    the core melody.

    Per docs: uploaded audio must be ≤ 8 minutes.
    """

    MODEL = "suno-upload-cover"
    CREATE_ENDPOINT = "/api/v1/generate/upload-cover"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "upload_url": ("STRING", {
                    "default": "",
                    "tooltip": "Public URL of audio to cover (≤ 8 min).",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Reimagine in a Lo-fi hip-hop style.",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
                "custom_mode": ("BOOLEAN", {"default": False}),
                "instrumental": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "style": ("STRING", {"default": ""}),
                "title": ("STRING", {"default": ""}),
                "negative_tags": ("STRING", {"default": ""}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": ""}),
                "style_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weirdness_constraint": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "persona_id": ("STRING", {"default": ""}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        upload = (kwargs.get("upload_url") or "").strip()
        if not upload:
            raise ValueError("Suno Upload And Cover requires upload_url.")
        body: dict[str, Any] = {
            "uploadUrl": upload,
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "customMode": bool(kwargs["custom_mode"]),
            "instrumental": bool(kwargs["instrumental"]),
        }
        for src, dst in [
            ("style", "style"), ("title", "title"),
            ("negative_tags", "negativeTags"), ("persona_id", "personaId"),
        ]:
            v = (kwargs.get(src) or "").strip()
            if v:
                body[dst] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


class SunoUploadAndExtend(BaseKieSunoMusicNode):
    """Upload an audio URL and extend it with AI-generated continuation."""

    MODEL = "suno-upload-extend"
    CREATE_ENDPOINT = "/api/v1/generate/upload-extend"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "upload_url": ("STRING", {
                    "default": "",
                    "tooltip": "Public URL of audio to extend.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Continue with a more energetic outro.",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
                "continue_at": ("INT", {"default": 30, "min": 0, "max": 600}),
            },
            "optional": {
                "instrumental": ("BOOLEAN", {"default": False}),
                "style": ("STRING", {"default": ""}),
                "title": ("STRING", {"default": ""}),
                "negative_tags": ("STRING", {"default": ""}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": ""}),
                "style_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weirdness_constraint": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
                "persona_id": ("STRING", {"default": ""}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        upload = (kwargs.get("upload_url") or "").strip()
        if not upload:
            raise ValueError("Suno Upload And Extend requires upload_url.")

        body: dict[str, Any] = {
            "uploadUrl": upload,
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "continueAt": int(kwargs["continue_at"]),
            "instrumental": bool(kwargs.get("instrumental", False)),
        }
        for src, dst in [
            ("style", "style"), ("title", "title"),
            ("negative_tags", "negativeTags"), ("persona_id", "personaId"),
        ]:
            v = (kwargs.get(src) or "").strip()
            if v:
                body[dst] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


class SunoMusicCover(BaseKieSunoMusicNode):
    """Generate a stylistic cover of a Suno-generated track.

    Distinct from UploadAndCover: this works on tracks already in
    Suno's system (via audio_id), not external uploads. Use this to
    quickly remix a track you just generated.
    """

    MODEL = "suno-music-cover"
    CREATE_ENDPOINT = "/api/v1/generate/cover-suno"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id of an existing Suno-generated track.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Reimagine as a jazz fusion track with saxophone solos.",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
            },
            "optional": {
                "style": ("STRING", {"default": ""}),
                "title": ("STRING", {"default": ""}),
                "negative_tags": ("STRING", {"default": ""}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": ""}),
                "style_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not audio_id:
            raise ValueError("Suno Music Cover requires audio_id.")

        body: dict[str, Any] = {
            "audioId": audio_id,
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
        }
        for src, dst in [
            ("style", "style"), ("title", "title"),
            ("negative_tags", "negativeTags"),
        ]:
            v = (kwargs.get(src) or "").strip()
            if v:
                body[dst] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        v = kwargs.get("style_weight")
        if v is not None:
            body["styleWeight"] = float(v)
        return body


# ============================================================ LAYERING

class SunoAddInstrumental(BaseKieSunoMusicNode):
    """Add an AI-generated instrumental backing to a vocal-only track.

    Per docs: uploadUrl is typically a vocal stem or melody recording.
    Output is a fully-arranged track with backing instruments.
    """

    MODEL = "suno-add-instrumental"
    CREATE_ENDPOINT = "/api/v1/generate/add-instrumental"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "upload_url": ("STRING", {
                    "default": "",
                    "tooltip": "URL of vocal stem or melody track.",
                }),
                "title": ("STRING", {
                    "default": "Untitled",
                    "tooltip": "Track title.",
                }),
                "tags": ("STRING", {
                    "default": "relaxing, piano, soothing",
                    "tooltip": "Style tags (comma-separated).",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5PLUS"}),
            },
            "optional": {
                "negative_tags": ("STRING", {"default": "heavy metal, fast drums"}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": ""}),
                "style_weight": ("FLOAT", {"default": 0.61, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weirdness_constraint": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        upload = (kwargs.get("upload_url") or "").strip()
        if not upload:
            raise ValueError("Suno Add Instrumental requires upload_url.")

        body: dict[str, Any] = {
            "uploadUrl": upload,
            "title": kwargs["title"],
            "tags": kwargs["tags"],
            "model": kwargs["model"],
        }
        v = (kwargs.get("negative_tags") or "").strip()
        if v:
            body["negativeTags"] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


class SunoAddVocals(BaseKieSunoMusicNode):
    """Add AI-generated vocals on top of an instrumental track.

    Per docs: provide prompt as lyrical concept + uploadUrl as instrumental.
    Style guides the vocal performance.
    """

    MODEL = "suno-add-vocals"
    CREATE_ENDPOINT = "/api/v1/generate/add-vocals"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "upload_url": ("STRING", {
                    "default": "",
                    "tooltip": "URL of instrumental track.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Soulful vocal layer about late-night reflection",
                }),
                "title": ("STRING", {
                    "default": "Vocal Layer",
                }),
                "style": ("STRING", {
                    "default": "Jazz",
                    "tooltip": "Vocal/genre style.",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5PLUS"}),
            },
            "optional": {
                "negative_tags": ("STRING", {"default": "heavy metal, aggressive vocals"}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": "m"}),
                "style_weight": ("FLOAT", {"default": 0.61, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weirdness_constraint": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        upload = (kwargs.get("upload_url") or "").strip()
        if not upload:
            raise ValueError("Suno Add Vocals requires upload_url.")

        body: dict[str, Any] = {
            "uploadUrl": upload,
            "prompt": kwargs["prompt"],
            "title": kwargs["title"],
            "style": kwargs["style"],
            "model": kwargs["model"],
        }
        v = (kwargs.get("negative_tags") or "").strip()
        if v:
            body["negativeTags"] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


# ============================================================ EDITING

class SunoBoostStyle(BaseKieSunoMusicNode):
    """Boost / reinforce the stylistic identity of a Suno track.

    Use when a generated track feels generic and needs more genre commitment.
    Schema inferred from naming patterns + Suno parameter conventions.
    """

    MODEL = "suno-boost-style"
    CREATE_ENDPOINT = "/api/v1/generate/boost-style"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id of the source Suno track to boost.",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
            },
            "optional": {
                "style": ("STRING", {
                    "default": "",
                    "tooltip": "Reinforce this style (e.g. 'lo-fi hip-hop').",
                }),
                "style_weight": ("FLOAT", {
                    "default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Higher = stronger style commitment.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not audio_id:
            raise ValueError("Suno Boost Style requires audio_id.")
        body: dict[str, Any] = {
            "audioId": audio_id,
            "model": kwargs["model"],
        }
        style = (kwargs.get("style") or "").strip()
        if style:
            body["style"] = style
        sw = kwargs.get("style_weight")
        if sw is not None:
            body["styleWeight"] = float(sw)
        return body


class SunoReplaceSection(BaseKieSunoMusicNode):
    """Replace a time-range section of a Suno track with new generated content.

    Re-generates between ``section_start`` and ``section_end`` (seconds),
    keeping the rest of the track unchanged. Great for fixing weak verses
    or replacing a bad chorus.
    """

    MODEL = "suno-replace-section"
    CREATE_ENDPOINT = "/api/v1/generate/replace-section"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id of source track.",
                }),
                "section_start": ("FLOAT", {
                    "default": 30.0, "min": 0.0, "max": 600.0, "step": 0.5,
                    "tooltip": "Section start (seconds).",
                }),
                "section_end": ("FLOAT", {
                    "default": 60.0, "min": 0.0, "max": 600.0, "step": 0.5,
                    "tooltip": "Section end (seconds).",
                }),
                "model": (_SUNO_MODELS, {"default": "V4_5"}),
            },
            "optional": {
                "lyric": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "New lyrics. Use [Verse]/[Chorus] markers.",
                }),
                "style": ("STRING", {
                    "default": "",
                    "tooltip": "Style for the replaced section.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not audio_id:
            raise ValueError("Suno Replace Section requires audio_id.")
        body: dict[str, Any] = {
            "audioId": audio_id,
            "replaceSectionStart": float(kwargs["section_start"]),
            "replaceSectionEnd": float(kwargs["section_end"]),
            "model": kwargs["model"],
        }
        for src, dst in [("lyric", "lyric"), ("style", "style")]:
            v = (kwargs.get(src) or "").strip()
            if v:
                body[dst] = v
        return body


# ============================================================ CREATIVE

class SunoGeneratePersona(BaseKieSunoMusicNode):
    """Create a Suno persona (reusable vocal/style identity) from a track.

    Per docs cURL: takes (taskId, audioId) of a source track and produces
    a persona ID that can be reused across future Generate Music calls
    via the ``persona_id`` field.

    Note: the output of this endpoint is a persona_id, NOT a new audio
    track. The base class will still return audio_id (= persona_id) in
    that slot for chaining. audio_path may be empty.
    """

    MODEL = "suno-generate-persona"
    CREATE_ENDPOINT = "/api/v1/generate/generate-persona"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "task_id": ("STRING", {
                    "default": "",
                    "tooltip": "task_id of the source Suno generation task.",
                }),
                "audio_id": ("STRING", {
                    "default": "",
                    "tooltip": "audio_id within that task.",
                }),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        task_id = (kwargs.get("task_id") or "").strip()
        audio_id = (kwargs.get("audio_id") or "").strip()
        if not task_id or not audio_id:
            raise ValueError("Suno Generate Persona requires both task_id and audio_id.")
        return {"taskId": task_id, "audioId": audio_id}


class SunoMashup(BaseKieSunoMusicNode):
    """Mashup two existing audio tracks into a coherent new composition.

    Per docs cURL: ``uploadUrlList`` is an array of up to 2 audio URLs.
    The model combines elements from each into a single track.
    """

    MODEL = "suno-mashup"
    CREATE_ENDPOINT = "/api/v1/generate/mashup"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "upload_urls": ("STRING", {
                    "default": "",
                    "tooltip": "Comma-separated 2 audio URLs to mash up.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "A seamless mashup blending the two tracks",
                }),
                "model": (_SUNO_MODELS, {"default": "V4"}),
                "custom_mode": ("BOOLEAN", {"default": True}),
                "instrumental": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "style": ("STRING", {"default": "Jazz"}),
                "title": ("STRING", {"default": "Mashup"}),
                "vocal_gender": (_VOCAL_GENDERS, {"default": ""}),
                "style_weight": ("FLOAT", {"default": 0.61, "min": 0.0, "max": 1.0, "step": 0.05}),
                "weirdness_constraint": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.05}),
                "audio_weight": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
        }

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        urls_csv = (kwargs.get("upload_urls") or "").strip()
        urls = [u.strip() for u in urls_csv.split(",") if u.strip()]
        if len(urls) < 2:
            raise ValueError(f"Suno Mashup requires exactly 2 upload_urls, got {len(urls)}.")
        if len(urls) > 2:
            raise ValueError(f"Suno Mashup accepts max 2 upload_urls, got {len(urls)}.")

        body: dict[str, Any] = {
            "uploadUrlList": urls,
            "prompt": kwargs["prompt"],
            "model": kwargs["model"],
            "customMode": bool(kwargs["custom_mode"]),
            "instrumental": bool(kwargs["instrumental"]),
        }
        for src, dst in [("style", "style"), ("title", "title")]:
            v = (kwargs.get(src) or "").strip()
            if v:
                body[dst] = v
        vg = kwargs.get("vocal_gender", "")
        if vg:
            body["vocalGender"] = vg
        for src, dst in [
            ("style_weight", "styleWeight"),
            ("weirdness_constraint", "weirdnessConstraint"),
            ("audio_weight", "audioWeight"),
        ]:
            v = kwargs.get(src)
            if v is not None:
                body[dst] = float(v)
        return body


# ----------------------------------------------------------------- Registration

NODE_CLASS_MAPPINGS: dict[str, type] = {
    "GenesisKieSunoGenerateMusic": SunoGenerateMusic,
    "GenesisKieSunoExtendMusic": SunoExtendMusic,
    "GenesisKieSunoUploadAndCover": SunoUploadAndCover,
    "GenesisKieSunoUploadAndExtend": SunoUploadAndExtend,
    "GenesisKieSunoMusicCover": SunoMusicCover,
    "GenesisKieSunoAddInstrumental": SunoAddInstrumental,
    "GenesisKieSunoAddVocals": SunoAddVocals,
    "GenesisKieSunoBoostStyle": SunoBoostStyle,
    "GenesisKieSunoReplaceSection": SunoReplaceSection,
    "GenesisKieSunoGeneratePersona": SunoGeneratePersona,
    "GenesisKieSunoMashup": SunoMashup,
}

NODE_DISPLAY_NAME_MAPPINGS: dict[str, str] = {
    "GenesisKieSunoGenerateMusic": "Kie — Suno Generate Music",
    "GenesisKieSunoExtendMusic": "Kie — Suno Extend Music",
    "GenesisKieSunoUploadAndCover": "Kie — Suno Upload And Cover",
    "GenesisKieSunoUploadAndExtend": "Kie — Suno Upload And Extend",
    "GenesisKieSunoMusicCover": "Kie — Suno Music Cover",
    "GenesisKieSunoAddInstrumental": "Kie — Suno Add Instrumental",
    "GenesisKieSunoAddVocals": "Kie — Suno Add Vocals",
    "GenesisKieSunoBoostStyle": "Kie — Suno Boost Style",
    "GenesisKieSunoReplaceSection": "Kie — Suno Replace Section",
    "GenesisKieSunoGeneratePersona": "Kie — Suno Generate Persona",
    "GenesisKieSunoMashup": "Kie — Suno Mashup",
}
