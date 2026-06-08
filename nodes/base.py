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


# ================================================ CHAT / LLM BASES (synchronous)

class BaseKieChatNode(BaseKieNode):
    """Abstract base for all chat / LLM nodes (synchronous, no polling).

    Common contract:
    - Inputs: ``system_prompt``, ``user_prompt``, optional ``image_url`` for
      multimodal models, ``max_tokens``, ``temperature``.
    - Outputs: ``(text, tokens_used)`` — the assistant's reply as STRING +
      total token usage as INT.

    Subclasses set:
    - ``ENDPOINT``: full Kie.ai path (e.g. "/gpt-5-2/v1/chat/completions").
    - ``MODEL_ID``: the model identifier sent in the body
      (e.g. "gpt-5-2", "claude-opus-4-8", "gemini-3.1-pro").
    - Override ``build_body()`` to assemble the family-specific request shape.
    - Override ``extract_output()`` to pull text + tokens from the response.

    Multimodal: if ``image_url`` is provided, subclasses inject it as an
    additional content part in the user message. Models that don't support
    images will return an error from Kie.ai (we don't pre-validate).
    """

    CATEGORY = CATEGORY_LLM
    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text", "tokens_used")

    ENDPOINT: ClassVar[str] = ""
    MODEL_ID: ClassVar[str] = ""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "user_prompt": ("STRING", {
                    "multiline": True,
                    "default": "Hello! Briefly introduce yourself.",
                }),
            },
            "optional": {
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "tooltip": "Optional system message (sets persona/context).",
                }),
                "image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional image URL for multimodal models.",
                }),
                "max_tokens": ("INT", {
                    "default": 2048, "min": 16, "max": 32768, "step": 16,
                    "tooltip": "Maximum tokens to generate.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Sampling temperature (0=deterministic, 1=normal, 2=very random).",
                }),
            },
        }

    def build_body(self, **kwargs: Any) -> dict[str, Any]:
        """Family-specific: build the request body. Override in subclass."""
        raise NotImplementedError

    def extract_output(self, response: dict[str, Any]) -> tuple[Any, ...]:
        """Family-specific: pull (text, tokens_used) from response. Override."""
        raise NotImplementedError

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        if not self.ENDPOINT:
            raise KieError(f"{type(self).__name__} did not declare ENDPOINT.")
        if not self.MODEL_ID:
            raise KieError(f"{type(self).__name__} did not declare MODEL_ID.")

        body = self.build_body(**kwargs)

        with KieClient() as client:
            response = client.chat_completion(self.ENDPOINT, body)

        return self.extract_output(response)


# ------------------------------------------------ Patrón A — OpenAI Chat Completions

class BaseKieChatOpenAINode(BaseKieChatNode):
    """For models that use the OpenAI ``/v1/chat/completions`` shape.

    Used by: GPT 5.2 and all Gemini variants (Gemini "openai" endpoints).

    Body:
        { messages: [{role, content}], temperature, max_tokens, ... }

    Response:
        { choices: [{ message: { content: "..." } }], usage: {...} }
    """

    def build_body(self, **kwargs: Any) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []

        system_prompt = (kwargs.get("system_prompt") or "").strip()
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        user_prompt = (kwargs.get("user_prompt") or "").strip()
        image_url = (kwargs.get("image_url") or "").strip()

        if image_url:
            # Multimodal content: text + image_url parts in user message.
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt or "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        body: dict[str, Any] = {
            "model": self.MODEL_ID,
            "messages": messages,
            "max_tokens": int(kwargs.get("max_tokens", 2048)),
            "temperature": float(kwargs.get("temperature", 0.7)),
            "stream": False,
        }
        return body

    def extract_output(self, response: dict[str, Any]) -> tuple[Any, ...]:
        choices = response.get("choices") or []
        if not choices:
            raise KieError(
                f"OpenAI chat response had no choices. Keys: {list(response.keys())}"
            )
        message = (choices[0] or {}).get("message") or {}
        content = message.get("content")

        # Content can be a plain string OR a list of content parts.
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict):
                    t = part.get("text") or part.get("content") or ""
                    if isinstance(t, str):
                        parts.append(t)
                elif isinstance(part, str):
                    parts.append(part)
            text = "".join(parts)

        if not text:
            raise KieError(
                f"OpenAI chat response had empty content. message: {message}"
            )

        usage = response.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        return (text, tokens)


# ------------------------------------------------ Patrón B — OpenAI Responses API

class BaseKieChatResponsesNode(BaseKieChatNode):
    """For models that use OpenAI's ``/v1/responses`` API shape.

    Used by: GPT 5.4, GPT 5.5, GPT Codex.

    Body:
        { model, input: [{role, content: [{type, text|image_url}]}],
          tools, reasoning: {effort}, stream }

    Response:
        { output: [
            {type: "reasoning", ...},
            {type: "message", content: [{type: "output_text", text: "..."}]}
          ],
          usage: {input_tokens, output_tokens, total_tokens},
          credits_consumed, status: "completed" }

    Extra input vs base:
    - ``reasoning_effort``: low/medium/high/xhigh — how deeply the model thinks.
    """

    REASONING_EFFORTS = ["minimal", "low", "medium", "high", "xhigh"]

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        types = super().INPUT_TYPES()
        types["optional"]["reasoning_effort"] = (cls.REASONING_EFFORTS, {
            "default": "medium",
            "tooltip": "How deeply the model should reason (higher = slower + more tokens).",
        })
        return types

    def build_body(self, **kwargs: Any) -> dict[str, Any]:
        input_array: list[dict[str, Any]] = []

        system_prompt = (kwargs.get("system_prompt") or "").strip()
        if system_prompt:
            # Responses API uses developer/system role similarly.
            input_array.append({
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            })

        user_prompt = (kwargs.get("user_prompt") or "").strip()
        image_url = (kwargs.get("image_url") or "").strip()

        content_parts: list[dict[str, Any]] = [
            {"type": "input_text", "text": user_prompt or "Describe the input."}
        ]
        if image_url:
            content_parts.append({
                "type": "input_image",
                "image_url": image_url,
            })

        input_array.append({"role": "user", "content": content_parts})

        body: dict[str, Any] = {
            "model": self.MODEL_ID,
            "input": input_array,
            "stream": False,
        }

        reasoning_effort = kwargs.get("reasoning_effort", "medium")
        if reasoning_effort:
            body["reasoning"] = {"effort": reasoning_effort}

        # NOTE: max_output_tokens is NOT included in the Responses API body.
        # Kie.ai's gateway returns {"code":500,"msg":"Server exception"} when
        # this field is present in /api/v1/responses requests (Codex endpoint).
        # The field is also non-standard in the Responses API spec — the model
        # uses its own default limit. The ``max_tokens`` input is kept for UI
        # consistency with Chat Completions nodes but is intentionally ignored
        # here. Verified via /home/claude debug script 2026-06.
        # If you need a hard cap, switch to a Chat Completions model (GPT 5.2
        # or any Gemini).
        return body

    def extract_output(self, response: dict[str, Any]) -> tuple[Any, ...]:
        outputs = response.get("output") or []
        if not outputs:
            raise KieError(
                f"Responses API: no output array. Keys: {list(response.keys())}"
            )

        # Find the first ``message`` block (skip reasoning blocks).
        text = ""
        for block in outputs:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "message":
                continue
            content = block.get("content") or []
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") in ("output_text", "text"):
                    text += part.get("text", "")
            if text:
                break

        if not text:
            # Fallback: dump the full output for debugging.
            import json
            raise KieError(
                f"Responses API: could not extract text. "
                f"output: {json.dumps(outputs, ensure_ascii=False)[:300]}"
            )

        usage = response.get("usage") or {}
        tokens = int(usage.get("total_tokens") or 0)
        return (text, tokens)


# ------------------------------------------------ Patrón C — Anthropic Messages

class BaseKieChatAnthropicNode(BaseKieChatNode):
    """For models that use Anthropic's ``/v1/messages`` shape.

    Used by: all Claude models (Opus 4.5-4.8, Sonnet 4.5-4.6, Haiku 4.5).

    Body:
        { model, messages: [{role, content}], max_tokens, temperature,
          system, thinking: {type, budget_tokens} }

    Response:
        { id, content: [{type: "text", text: "..."}], usage: {...} }

    Extra input vs base:
    - ``thinking``: enables extended reasoning ("thinking blocks") on Claude.
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        types = super().INPUT_TYPES()
        types["optional"]["thinking"] = ("BOOLEAN", {
            "default": False,
            "tooltip": "Enable Claude's extended thinking (more accurate, slower).",
        })
        types["optional"]["thinking_budget"] = ("INT", {
            "default": 4096, "min": 1024, "max": 32000, "step": 256,
            "tooltip": "Token budget for thinking blocks (only used if thinking=true).",
        })
        return types

    def build_body(self, **kwargs: Any) -> dict[str, Any]:
        messages: list[dict[str, Any]] = []

        user_prompt = (kwargs.get("user_prompt") or "").strip()
        image_url = (kwargs.get("image_url") or "").strip()

        if image_url:
            # Anthropic image format expects ``source`` with type=url.
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt or "Describe this image."},
                    {
                        "type": "image",
                        "source": {"type": "url", "url": image_url},
                    },
                ],
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        body: dict[str, Any] = {
            "model": self.MODEL_ID,
            "messages": messages,
            "max_tokens": int(kwargs.get("max_tokens", 2048)),
            "stream": False,
        }

        # System prompt is its own field in Anthropic, not a message.
        system_prompt = (kwargs.get("system_prompt") or "").strip()
        if system_prompt:
            body["system"] = system_prompt

        # Temperature only matters if thinking is OFF.
        thinking = bool(kwargs.get("thinking", False))
        if thinking:
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": int(kwargs.get("thinking_budget", 4096)),
            }
            # When thinking is on, temperature must be 1.0 (Anthropic constraint).
            body["temperature"] = 1.0
        else:
            body["temperature"] = float(kwargs.get("temperature", 0.7))

        return body

    def extract_output(self, response: dict[str, Any]) -> tuple[Any, ...]:
        content = response.get("content") or []
        if not content:
            raise KieError(
                f"Anthropic response had no content. Keys: {list(response.keys())}"
            )

        # Find the first text block (skip thinking blocks).
        text = ""
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text += block.get("text", "")

        if not text:
            import json
            raise KieError(
                f"Anthropic response: could not extract text. "
                f"content: {json.dumps(content, ensure_ascii=False)[:300]}"
            )

        usage = response.get("usage") or {}
        # Anthropic uses input_tokens + output_tokens (no total_tokens).
        tokens = int(
            (usage.get("input_tokens") or 0)
            + (usage.get("output_tokens") or 0)
        )
        return (text, tokens)
