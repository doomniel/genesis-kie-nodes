"""Base classes for all Genesis-Kie ComfyUI nodes.

Two distinct base classes for video models, matching Kie.ai's two endpoint
patterns:

- :class:`BaseKieMarketVideoNode` — for Seedance, Kling, Hailuo, Wan,
  HappyHorse, Grok, ElevenLabs, etc. Uses ``/api/v1/jobs/createTask`` and
  ``/api/v1/jobs/recordInfo``.

- :class:`BaseKieVeoVideoNode` — for Veo 3.1 only. Uses
  ``/api/v1/veo/generate`` and ``/api/v1/veo/record-info`` with a different
  response shape.

Subclasses only declare:

- The Kie ``MODEL`` identifier
- The ComfyUI ``INPUT_TYPES`` shape
- ``build_input(**kwargs)`` to map ComfyUI inputs → Kie input dict
- (Rarely) override ``extract_output`` for unusual response shapes

Everything else (HTTP, polling, JSON parsing, downloading, error mapping) is
handled here.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..client import (
    KieClient,
    KieError,
    all_urls,
    download_to_output,
    first_url,
    image_url_to_tensor,
    images_urls_to_tensor,
)

log = logging.getLogger("genesis_kie")


# Category strings shown in the ComfyUI menu.
CATEGORY_VIDEO = "GenesisKie/Video"
CATEGORY_IMAGE = "GenesisKie/Image"
CATEGORY_MUSIC = "GenesisKie/Music"
CATEGORY_LLM = "GenesisKie/LLM"


# ----------------------------------------------------------------- common base

class BaseKieNode:
    """Common ComfyUI plumbing for any Kie-backed node.

    Concrete subclasses (or modality-specific bases like
    :class:`BaseKieMarketVideoNode`) take care of HTTP + parsing.
    """

    MODEL: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = "GenesisKie"
    RETURN_TYPES: ClassVar[tuple[str, ...]] = ()
    RETURN_NAMES: ClassVar[tuple[str, ...]] = ()
    FUNCTION: ClassVar[str] = "run"

    # Polling defaults — subclasses may override per-model.
    POLL_INTERVAL_SECONDS: ClassVar[float] = 3.0
    TIMEOUT_SECONDS: ClassVar[float] = 600.0

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI inputs into the Kie ``input`` dict.

        Default implementation drops ``None`` and empty strings. Override
        when you need to rename keys, coerce types, or drop empty values.
        """
        return {k: v for k, v in kwargs.items() if v not in (None, "")}

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        """Turn the Kie ``data`` dict into the ComfyUI return tuple."""
        raise NotImplementedError


# ================================================== MARKET (jobs/createTask)

class BaseKieMarketVideoNode(BaseKieNode):
    """Base class for video nodes that use the Market generic endpoint.

    Covers: Seedance, Kling (except Avatar/motion-control may extend),
    Hailuo, Wan, HappyHorse, Grok Imagine, Runway, etc.

    Subclasses MUST set ``MODEL`` and define ``INPUT_TYPES``. They typically
    override ``build_input`` to translate ComfyUI inputs to the model's
    specific ``input`` schema.
    """

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    # Video usually takes longer than image. 15 min upper bound.
    TIMEOUT_SECONDS = 900.0
    POLL_INTERVAL_SECONDS = 5.0

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.MODEL:
            raise KieError(f"{type(self).__name__} did not declare MODEL.")

        inputs = self.build_input(**kwargs)
        log.debug(
            "Kie market video node=%s model=%s inputs=%s",
            type(self).__name__, self.MODEL, inputs,
        )

        with KieClient() as client:
            last_state = [None]

            def on_progress(d: dict[str, Any]) -> None:
                state = d.get("state")
                if state != last_state[0]:
                    log.info("Kie task state=%s", state)
                    last_state[0] = state

            data = client.run_market(
                self.MODEL,
                inputs,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
                progress_callback=on_progress,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        url = first_url(data)
        if not url:
            parsed = data.get("_parsed_result") or {}
            raise KieError(
                "Kie returned no video URL. "
                f"_parsed_result keys: "
                f"{list(parsed.keys()) if isinstance(parsed, dict) else 'n/a'}, "
                f"data keys: {list(data.keys())}"
            )
        path = download_to_output(url, prefix="kie_video", fallback_ext="mp4")
        return (path,)


class BaseKieMarketImageNode(BaseKieNode):
    """Base for image nodes using the Market endpoint.

    Returns a ComfyUI ``IMAGE`` tensor of shape ``(B, H, W, C)``.
    When the API returns multiple images (e.g. Seedream 4.0 with
    ``max_images=4``, Wan 2.7 Image with ``n=4``, Ideogram with
    ``num_images>1``), they are stacked into a single batch tensor.

    All images are resized to the smallest common (W, H) to allow batch
    stacking. If you need original sizes, subclass and override
    :meth:`extract_output` to return only the first URL.
    """

    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TIMEOUT_SECONDS = 300.0
    POLL_INTERVAL_SECONDS = 2.0

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        if not self.MODEL:
            raise KieError(f"{type(self).__name__} did not declare MODEL.")

        inputs = self.build_input(**kwargs)
        log.debug(
            "Kie market image node=%s model=%s inputs=%s",
            type(self).__name__, self.MODEL, inputs,
        )

        with KieClient() as client:
            data = client.run_market(
                self.MODEL,
                inputs,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        urls = all_urls(data)
        if not urls:
            # Fallback to first_url in case all_urls returned empty.
            single = first_url(data)
            if single:
                urls = [single]
        if not urls:
            parsed = data.get("_parsed_result") or {}
            raise KieError(
                "Kie returned no image URL. "
                f"_parsed_result keys: "
                f"{list(parsed.keys()) if isinstance(parsed, dict) else 'n/a'}, "
                f"data keys: {list(data.keys())}"
            )
        log.info("Kie image URLs: %d returned", len(urls))
        tensor = images_urls_to_tensor(urls)
        return (tensor,)


class BaseKieMarketAudioNode(BaseKieNode):
    """Base for audio/music nodes using the Market endpoint."""

    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)

    TIMEOUT_SECONDS = 600.0
    POLL_INTERVAL_SECONDS = 3.0

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.MODEL:
            raise KieError(f"{type(self).__name__} did not declare MODEL.")
        inputs = self.build_input(**kwargs)
        with KieClient() as client:
            data = client.run_market(
                self.MODEL,
                inputs,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )
        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        url = first_url(data)
        if not url:
            raise KieError(
                f"Kie returned no audio URL. data keys: {list(data.keys())}"
            )
        path = download_to_output(url, prefix="kie_audio", fallback_ext="mp3")
        return (path,)


class BaseKieMarketTextNode(BaseKieNode):
    """Base for chat/LLM nodes using the Market endpoint."""

    CATEGORY = CATEGORY_LLM
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    TIMEOUT_SECONDS = 180.0
    POLL_INTERVAL_SECONDS = 1.0

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.MODEL:
            raise KieError(f"{type(self).__name__} did not declare MODEL.")
        inputs = self.build_input(**kwargs)
        with KieClient() as client:
            data = client.run_market(
                self.MODEL,
                inputs,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )
        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        parsed = data.get("_parsed_result") or {}
        if isinstance(parsed, dict):
            obj = parsed.get("resultObject") or parsed
            if isinstance(obj, dict):
                # Common shapes: {"content": "..."}, {"text": "..."},
                # OpenAI-style {"choices": [{"message": {"content": "..."}}]}
                for key in ("content", "text", "message", "output"):
                    val = obj.get(key)
                    if isinstance(val, str) and val:
                        return (val,)
                    if isinstance(val, dict):
                        sub = val.get("content") or val.get("text")
                        if isinstance(sub, str) and sub:
                            return (sub,)
                choices = obj.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str):
                            return (content,)
            if isinstance(obj, str):
                return (obj,)

        raise KieError(
            f"Kie returned no text content. "
            f"_parsed_result keys: "
            f"{list(parsed.keys()) if isinstance(parsed, dict) else 'n/a'}, "
            f"data keys: {list(data.keys())}"
        )


# ======================================================= VEO (dedicated API)

class BaseKieVeoVideoNode(BaseKieNode):
    """Base class for Veo 3.1 video nodes (dedicated endpoint).

    The Veo API has a different shape than the Market API:
    - Request goes to ``/api/v1/veo/generate`` with parameters at the top
      level (NOT wrapped in ``input``).
    - Status uses ``successFlag`` (0/1/2/3) instead of ``state``.
    - URLs live at ``data.response.resultUrls`` (a real array, not a
      JSON string).
    """

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    TIMEOUT_SECONDS = 900.0
    POLL_INTERVAL_SECONDS = 5.0

    def build_veo_request(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI inputs into kwargs for ``client.run_veo()``.

        Override to customize per-tier behavior. The default expects ComfyUI
        kwargs that match ``run_veo``'s parameters directly.
        """
        return kwargs

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.MODEL:
            raise KieError(f"{type(self).__name__} did not declare MODEL.")

        veo_kwargs = self.build_veo_request(**kwargs)
        veo_kwargs.setdefault("model", self.MODEL)

        with KieClient() as client:
            last_flag = [None]

            def on_progress(d: dict[str, Any]) -> None:
                flag = d.get("successFlag")
                if flag != last_flag[0]:
                    log.info("Veo successFlag=%s", flag)
                    last_flag[0] = flag

            data = client.run_veo(
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
                progress_callback=on_progress,
                **veo_kwargs,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        response = data.get("response") or {}
        urls = response.get("resultUrls") or []
        if not urls or not isinstance(urls, list):
            raise KieError(
                "Veo response did not include resultUrls. "
                f"response keys: {list(response.keys())}, "
                f"data keys: {list(data.keys())}"
            )
        video_url = urls[0]
        if not isinstance(video_url, str) or not video_url:
            raise KieError(f"Empty video URL in resultUrls: {urls}")
        log.info("Veo video URL: %s", video_url)
        path = download_to_output(video_url, prefix="kie_veo", fallback_ext="mp4")
        return (path,)


# ================================================== 4O IMAGE (dedicated API)

class BaseKie4oImageNode(BaseKieNode):
    """Base for OpenAI 4o Image nodes (dedicated endpoint).

    Like Veo, this uses a top-level camelCase body (no ``input`` wrapper)
    and ``data.successFlag`` for status. Result URLs live at
    ``data.response.resultUrls``.

    Subclasses set how many variants are requested and override
    :meth:`build_4o_request` to populate ``run_4o_image`` kwargs.
    """

    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TIMEOUT_SECONDS = 600.0
    POLL_INTERVAL_SECONDS = 3.0

    def build_4o_request(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI inputs into kwargs for ``client.run_4o_image()``.

        Default expects ComfyUI kwargs match ``run_4o_image``'s parameters.
        Override to map or rename fields.
        """
        return kwargs

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        request_kwargs = self.build_4o_request(**kwargs)

        with KieClient() as client:
            last_flag = [None]

            def on_progress(d: dict[str, Any]) -> None:
                flag = d.get("successFlag")
                if flag != last_flag[0]:
                    log.info("4o image successFlag=%s", flag)
                    last_flag[0] = flag

            data = client.run_4o_image(
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
                progress_callback=on_progress,
                **request_kwargs,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        urls = all_urls(data)
        if not urls:
            raise KieError(
                "4o image returned no resultUrls. "
                f"response keys: {list((data.get('response') or {}).keys())}, "
                f"data keys: {list(data.keys())}"
            )
        log.info("4o image URLs: %d returned", len(urls))
        tensor = images_urls_to_tensor(urls)
        return (tensor,)


# =============================================== FLUX KONTEXT (dedicated API)

class BaseKieFluxKontextNode(BaseKieNode):
    """Base for Flux Kontext nodes (dedicated endpoint).

    Like 4o Image / Veo: top-level camelCase body, ``successFlag`` status,
    ``response.resultUrls`` result.

    Subclasses set ``MODEL`` (``flux-kontext-pro`` or ``flux-kontext-max``)
    and override :meth:`build_kontext_request` if needed.
    """

    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    TIMEOUT_SECONDS = 600.0
    POLL_INTERVAL_SECONDS = 3.0

    def build_kontext_request(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI inputs into kwargs for ``client.run_flux_kontext()``."""
        return kwargs

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        if not self.MODEL:
            raise KieError(f"{type(self).__name__} did not declare MODEL.")

        request_kwargs = self.build_kontext_request(**kwargs)
        request_kwargs.setdefault("model", self.MODEL)

        with KieClient() as client:
            last_flag = [None]

            def on_progress(d: dict[str, Any]) -> None:
                flag = d.get("successFlag")
                if flag != last_flag[0]:
                    log.info("Flux kontext successFlag=%s", flag)
                    last_flag[0] = flag

            data = client.run_flux_kontext(
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
                progress_callback=on_progress,
                **request_kwargs,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        urls = all_urls(data)
        if not urls:
            raise KieError(
                "Flux kontext returned no resultUrls. "
                f"response keys: {list((data.get('response') or {}).keys())}, "
                f"data keys: {list(data.keys())}"
            )
        log.info("Flux kontext URLs: %d returned", len(urls))
        tensor = images_urls_to_tensor(urls)
        return (tensor,)


# =============================================== SUNO MUSIC (dedicated API)

class BaseKieSunoMusicNode(BaseKieNode):
    """Base for Suno music-generation nodes (dedicated endpoint).

    These nodes return THREE outputs:

    - ``audio_path`` (STRING): local path to the FIRST audio variant.
      Connect to a Save/Play node downstream.
    - ``audio_id`` (STRING): Suno's internal ID for the first variant.
      Required to chain into Extend / AddVocals / Cover / Replace nodes.
    - ``all_paths_csv`` (STRING): comma-separated paths of ALL variants
      (Suno typically returns 2-4 candidates per request).

    Subclasses set:
    - ``CREATE_ENDPOINT``: the POST URL (e.g. "/api/v1/generate").
    - ``POLLING_ENDPOINT``: GET URL (typically "/api/v1/generate/record-info").
    - Override ``build_suno_body()`` to map ComfyUI inputs → request body.
    """

    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("audio_path", "audio_id", "all_paths_csv")

    CREATE_ENDPOINT: ClassVar[str] = ""
    POLLING_ENDPOINT: ClassVar[str] = "/api/v1/generate/record-info"

    TIMEOUT_SECONDS = 600.0
    POLL_INTERVAL_SECONDS = 5.0

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        """Translate ComfyUI inputs into the Suno request body."""
        return {k: v for k, v in kwargs.items() if v not in (None, "")}

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.CREATE_ENDPOINT:
            raise KieError(f"{type(self).__name__} did not declare CREATE_ENDPOINT.")

        body = self.build_suno_body(**kwargs)

        with KieClient() as client:
            last_status = [None]

            def on_progress(d: dict[str, Any]) -> None:
                status = d.get("status")
                if status != last_status[0]:
                    log.info("Suno status=%s", status)
                    last_status[0] = status

            data = client.run_suno_task(
                self.CREATE_ENDPOINT,
                self.POLLING_ENDPOINT,
                body,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
                progress_callback=on_progress,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        """Pull audio URLs from ``data.response.sunoData`` and download them."""
        response = data.get("response") or {}
        suno_data = response.get("sunoData") or []

        if not suno_data:
            raise KieError(
                "Suno task returned no sunoData. "
                f"response keys: {list(response.keys())}, "
                f"data keys: {list(data.keys())}"
            )

        # Suno typically returns 2-4 audio candidates.
        paths: list[str] = []
        ids: list[str] = []
        for item in suno_data:
            if not isinstance(item, dict):
                continue
            url = item.get("audioUrl")
            audio_id = item.get("id")
            if not url:
                continue
            path = download_to_output(url, prefix="kie_suno", fallback_ext="mp3")
            paths.append(path)
            if audio_id:
                ids.append(audio_id)

        if not paths:
            raise KieError(f"Suno sunoData had no audioUrl entries. sunoData: {suno_data}")

        log.info("Suno returned %d variant(s)", len(paths))
        first_path = paths[0]
        first_id = ids[0] if ids else ""
        all_paths_csv = ",".join(paths)
        return (first_path, first_id, all_paths_csv)


# ============================================= SUNO TEXT (Lyrics / Timestamped)

class BaseKieSunoTextNode(BaseKieNode):
    """Base for Suno endpoints that return TEXT (lyrics, timestamps).

    Output:
    - ``text`` (STRING): the lyrics or JSON-stringified timestamps.

    Subclasses set CREATE_ENDPOINT + POLLING_ENDPOINT (or override run()
    for synchronous endpoints).
    """

    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    CREATE_ENDPOINT: ClassVar[str] = ""
    POLLING_ENDPOINT: ClassVar[str] = ""

    TIMEOUT_SECONDS = 180.0
    POLL_INTERVAL_SECONDS = 2.0

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v not in (None, "")}

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.CREATE_ENDPOINT:
            raise KieError(f"{type(self).__name__} did not declare CREATE_ENDPOINT.")

        body = self.build_suno_body(**kwargs)

        with KieClient() as client:
            data = client.run_suno_task(
                self.CREATE_ENDPOINT,
                self.POLLING_ENDPOINT,
                body,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        """Subclasses customize per endpoint shape."""
        response = data.get("response") or {}
        # Try common field names
        for key in ("lyrics", "text", "lyricsData", "timestampedLyrics"):
            value = response.get(key)
            if isinstance(value, str) and value:
                return (value,)
            if isinstance(value, (list, dict)):
                import json
                return (json.dumps(value, ensure_ascii=False),)
        raise KieError(
            f"Suno text task returned no usable text. response keys: "
            f"{list(response.keys())}"
        )


# ============================================= SUNO AUDIO UTILITY (WAV/Stem/MIDI)

class BaseKieSunoAudioUtilityNode(BaseKieNode):
    """Base for Suno audio-utility endpoints (WAV convert, MIDI, etc).

    Returns a single audio/midi file path.

    Subclasses set:
    - ``CREATE_ENDPOINT``: e.g. "/api/v1/wav/generate".
    - ``POLLING_ENDPOINT``: e.g. "/api/v1/wav/record-info".
    - ``OUTPUT_EXT``: file extension hint ("wav", "mid", "mp3").
    - ``RESULT_FIELD``: response field with the URL (typically
      "audioUrl" or "midiUrl").
    """

    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)

    CREATE_ENDPOINT: ClassVar[str] = ""
    POLLING_ENDPOINT: ClassVar[str] = ""
    OUTPUT_EXT: ClassVar[str] = "mp3"
    RESULT_FIELD: ClassVar[str] = "audioUrl"

    TIMEOUT_SECONDS = 300.0
    POLL_INTERVAL_SECONDS = 3.0

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v not in (None, "")}

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        if not self.CREATE_ENDPOINT:
            raise KieError(f"{type(self).__name__} did not declare CREATE_ENDPOINT.")

        body = self.build_suno_body(**kwargs)

        with KieClient() as client:
            data = client.run_suno_task(
                self.CREATE_ENDPOINT,
                self.POLLING_ENDPOINT,
                body,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        response = data.get("response") or {}
        url = response.get(self.RESULT_FIELD)
        if not url:
            # Try nested arrays (e.g. vocal removal returns multiple stems)
            for key in ("audioUrls", "audios", "stems", "results"):
                arr = response.get(key)
                if isinstance(arr, list) and arr:
                    first = arr[0]
                    if isinstance(first, str):
                        url = first
                        break
                    if isinstance(first, dict):
                        url = first.get("url") or first.get("audioUrl")
                        if url:
                            break
        if not url:
            raise KieError(
                f"Suno audio utility task: no '{self.RESULT_FIELD}' URL found. "
                f"response keys: {list(response.keys())}"
            )
        path = download_to_output(url, prefix="kie_suno_util", fallback_ext=self.OUTPUT_EXT)
        return (path,)


# =============================================== SUNO STEM SEPARATION (multi-output)

class BaseKieSunoStemSeparationNode(BaseKieNode):
    """Base for Suno Vocal Removal / Stem Separation (multi-stem output).

    Vocal removal returns multiple stems (vocal + instrumental, or
    full multi-instrument separation depending on ``type`` param).

    Outputs:
    - ``vocals_path`` (STRING): isolated vocals stem
    - ``instrumental_path`` (STRING): instrumental (no-vocals) stem
    - ``all_stems_csv`` (STRING): comma-separated paths of all returned stems
    """

    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("vocals_path", "instrumental_path", "all_stems_csv")

    CREATE_ENDPOINT: ClassVar[str] = "/api/v1/vocal-removal/generate"
    POLLING_ENDPOINT: ClassVar[str] = "/api/v1/vocal-removal/record-info"

    TIMEOUT_SECONDS = 300.0
    POLL_INTERVAL_SECONDS = 3.0

    def build_suno_body(self, **kwargs: Any) -> dict[str, Any]:
        return {k: v for k, v in kwargs.items() if v not in (None, "")}

    def run(self, **kwargs: Any) -> tuple[str, ...]:
        body = self.build_suno_body(**kwargs)

        with KieClient() as client:
            data = client.run_suno_task(
                self.CREATE_ENDPOINT,
                self.POLLING_ENDPOINT,
                body,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )

        return self.extract_output(data)

    def extract_output(self, data: dict[str, Any]) -> tuple[str, ...]:
        response = data.get("response") or {}

        # Try named keys first (the documented shape may use vocalUrl/instrumentalUrl).
        vocals_url = response.get("vocalUrl") or response.get("vocalsUrl")
        instr_url = response.get("instrumentalUrl") or response.get("noVocalsUrl")
        all_urls: list[str] = []

        if vocals_url:
            all_urls.append(vocals_url)
        if instr_url:
            all_urls.append(instr_url)

        # Try generic arrays.
        if not all_urls:
            for key in ("stems", "audioUrls", "results"):
                arr = response.get(key)
                if isinstance(arr, list):
                    for item in arr:
                        if isinstance(item, str):
                            all_urls.append(item)
                        elif isinstance(item, dict):
                            url = item.get("url") or item.get("audioUrl")
                            if url:
                                all_urls.append(url)
                    if all_urls:
                        break

        if not all_urls:
            raise KieError(
                f"Vocal removal task returned no stem URLs. "
                f"response keys: {list(response.keys())}"
            )

        # Download each stem
        paths: list[str] = []
        for url in all_urls:
            path = download_to_output(url, prefix="kie_stem", fallback_ext="mp3")
            paths.append(path)

        vocals_path = paths[0] if vocals_url else (paths[0] if paths else "")
        instrumental_path = paths[1] if instr_url and len(paths) >= 2 else (
            paths[1] if len(paths) >= 2 else ""
        )
        all_stems_csv = ",".join(paths)

        log.info("Vocal removal returned %d stems", len(paths))
        return (vocals_path, instrumental_path, all_stems_csv)
