"""Base classes for all Genesis-Kie ComfyUI nodes.

The intent: each concrete node should only need to declare:

- The Kie ``model`` identifier
- The ComfyUI ``INPUT_TYPES`` shape
- How to map ComfyUI inputs to the Kie ``input`` dict
- Where the result URL lives in the Kie ``data`` dict

Everything else (HTTP, polling, downloading, error mapping) is handled here.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from ..client import (
    KieClient,
    KieError,
    download_to_output,
    first_url,
    image_url_to_tensor,
)

log = logging.getLogger("genesis_kie")


# Category strings shown in the ComfyUI menu. Keeping them short and grouped
# under a single "GenesisKie" prefix so users can find them easily.
CATEGORY_VIDEO = "GenesisKie/Video"
CATEGORY_IMAGE = "GenesisKie/Image"
CATEGORY_MUSIC = "GenesisKie/Music"
CATEGORY_LLM = "GenesisKie/LLM"


class BaseKieNode:
    """Common scaffolding for any node that talks to Kie.ai.

    Subclasses must define:

    - ``MODEL_ID``: str — Kie.ai model identifier (e.g. ``"google/veo-3.1-fast"``).
    - ``INPUT_TYPES``: classmethod returning the ComfyUI input schema.
    - ``RETURN_TYPES``: tuple of output type names.
    - ``FUNCTION``: name of the method to call (typically ``"run"``).
    - ``build_input(...)``: instance method that maps ComfyUI inputs to the
      Kie ``input`` dict.
    - ``extract_output(data: dict)``: instance method that turns the Kie
      result into the ComfyUI return tuple.
    """

    # Subclasses MUST override these.
    MODEL_ID: ClassVar[str] = ""
    CATEGORY: ClassVar[str] = "GenesisKie"
    RETURN_TYPES: ClassVar[tuple[str, ...]] = ()
    RETURN_NAMES: ClassVar[tuple[str, ...]] = ()
    FUNCTION: ClassVar[str] = "run"

    # Polling defaults — subclasses may override (video tends to take longer).
    POLL_INTERVAL_SECONDS: ClassVar[float] = 3.0
    TIMEOUT_SECONDS: ClassVar[float] = 600.0

    # ---------------------------------------------------------- subclass API

    def build_input(self, **kwargs: Any) -> dict[str, Any]:
        """Translate the ComfyUI inputs into the Kie ``input`` dict.

        Default implementation passes everything straight through. Override
        when you need to rename keys, coerce types, or drop empty values.
        """
        return {k: v for k, v in kwargs.items() if v not in (None, "")}

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        """Turn the Kie ``data`` dict into the ComfyUI return tuple.

        Must be overridden by concrete subclasses.
        """
        raise NotImplementedError

    # ---------------------------------------------------------- ComfyUI hook

    def run(self, **kwargs: Any) -> tuple[Any, ...]:
        if not self.MODEL_ID:
            raise KieError(
                f"{type(self).__name__} did not declare MODEL_ID."
            )

        inputs = self.build_input(**kwargs)
        log.debug("Kie node=%s model=%s inputs=%s",
                  type(self).__name__, self.MODEL_ID, inputs)

        with KieClient() as client:
            data = client.run(
                self.MODEL_ID,
                inputs,
                poll_interval=self.POLL_INTERVAL_SECONDS,
                timeout=self.TIMEOUT_SECONDS,
            )

        return self.extract_output(data)


class BaseKieVideoNode(BaseKieNode):
    """Base class for video-generation nodes.

    Returns a single output: the absolute path to the downloaded video file
    in ComfyUI's output dir. Most video models in Kie return ``video_url``
    or ``output.video_url``.
    """

    CATEGORY = CATEGORY_VIDEO
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)

    # Keys to check for the result URL, in priority order.
    URL_KEYS: ClassVar[list[str]] = [
        "video_url",
        "videoUrl",
        "video",
        "output_video_url",
        "url",
        "output",
        "result",
    ]

    # Wait up to 15 minutes for video by default — some models (Veo Quality
    # 4K, Sora Pro) routinely take 5-10 min.
    TIMEOUT_SECONDS = 900.0
    POLL_INTERVAL_SECONDS = 5.0

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        url = first_url(data, self.URL_KEYS)
        if not url:
            raise KieError(
                f"Kie returned no video URL. Keys in data: {list(data.keys())}"
            )
        path = download_to_output(url, prefix="kie_video", fallback_ext="mp4")
        return (path,)


class BaseKieImageNode(BaseKieNode):
    """Base class for image-generation nodes.

    Returns a single ComfyUI ``IMAGE`` tensor (BHWC float32 in [0, 1]).
    """

    CATEGORY = CATEGORY_IMAGE
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)

    URL_KEYS: ClassVar[list[str]] = [
        "image_url",
        "imageUrl",
        "image",
        "output_image_url",
        "url",
        "images",
        "output",
        "result",
    ]

    TIMEOUT_SECONDS = 300.0
    POLL_INTERVAL_SECONDS = 2.0

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        url = first_url(data, self.URL_KEYS)
        if not url:
            raise KieError(
                f"Kie returned no image URL. Keys in data: {list(data.keys())}"
            )
        tensor = image_url_to_tensor(url)
        return (tensor,)


class BaseKieAudioNode(BaseKieNode):
    """Base class for audio / music generation nodes.

    Returns the absolute path to the downloaded audio file.
    """

    CATEGORY = CATEGORY_MUSIC
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("audio_path",)

    URL_KEYS: ClassVar[list[str]] = [
        "audio_url",
        "audioUrl",
        "audio",
        "music_url",
        "url",
        "output",
        "result",
    ]

    TIMEOUT_SECONDS = 600.0
    POLL_INTERVAL_SECONDS = 3.0

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        url = first_url(data, self.URL_KEYS)
        if not url:
            raise KieError(
                f"Kie returned no audio URL. Keys in data: {list(data.keys())}"
            )
        path = download_to_output(url, prefix="kie_audio", fallback_ext="mp3")
        return (path,)


class BaseKieTextNode(BaseKieNode):
    """Base class for chat / LLM nodes.

    Returns the assistant's text response.
    """

    CATEGORY = CATEGORY_LLM
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)

    TIMEOUT_SECONDS = 120.0
    POLL_INTERVAL_SECONDS = 1.0

    def extract_output(self, data: dict[str, Any]) -> tuple[Any, ...]:
        # LLM endpoints typically return text under ``content`` or ``output``.
        text = data.get("content") or data.get("output") or data.get("text")
        if isinstance(text, dict):
            text = text.get("text") or text.get("content")
        if not isinstance(text, str):
            raise KieError(
                f"Kie returned no text content. Keys in data: {list(data.keys())}"
            )
        return (text,)
