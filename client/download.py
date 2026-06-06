"""Helpers for downloading Kie.ai result URLs into local files / tensors.

ComfyUI nodes consume tensors for images and file paths for videos. This
module bridges the gap by:

- Downloading the URL bytes via httpx.
- For images: decoding to a torch tensor in the ComfyUI BHWC float32 format.
- For videos / audio: saving to ComfyUI's ``output/`` directory and returning
  the absolute path.

We keep torch / numpy / PIL imports lazy so this package can still be
imported in environments where ComfyUI is not present (e.g. tests).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from .exceptions import KieError

log = logging.getLogger("genesis_kie")

_DOWNLOAD_TIMEOUT_SECONDS = 120.0
_CHUNK_SIZE = 64 * 1024


def _comfyui_output_dir() -> Path:
    """Resolve ComfyUI's output directory.

    Tries ``folder_paths.get_output_directory()`` first (the official ComfyUI
    API). Falls back to ``./output`` relative to the current working dir.
    """
    try:
        import folder_paths  # type: ignore[import-not-found]
        return Path(folder_paths.get_output_directory())
    except Exception:  # noqa: BLE001
        return Path.cwd() / "output"


def _safe_filename(url: str, fallback_ext: str = "bin") -> str:
    parsed = urlparse(url)
    name = os.path.basename(parsed.path) or f"kie_output.{fallback_ext}"
    # Strip query strings / unsafe chars
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if "." not in name:
        name = f"{name}.{fallback_ext}"
    return name


def download_bytes(url: str, *, timeout: float = _DOWNLOAD_TIMEOUT_SECONDS) -> bytes:
    """Fetch the URL and return its raw bytes."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as exc:
        raise KieError(f"Failed to download {url}: {exc}") from exc


def download_to_output(
    url: str,
    *,
    prefix: str = "kie",
    fallback_ext: str = "bin",
    timeout: float = _DOWNLOAD_TIMEOUT_SECONDS,
) -> str:
    """Download ``url`` into ComfyUI's output dir and return the absolute path.

    The filename is derived from the URL with a unique numeric suffix to avoid
    collisions across runs.
    """
    out_dir = _comfyui_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_name = _safe_filename(url, fallback_ext=fallback_ext)
    stem, dot, ext = base_name.rpartition(".")
    if not dot:
        stem, ext = base_name, fallback_ext

    # Find a non-conflicting filename.
    counter = 0
    while True:
        candidate = f"{prefix}_{stem}_{counter:04d}.{ext}" if counter else f"{prefix}_{stem}.{ext}"
        path = out_dir / candidate
        if not path.exists():
            break
        counter += 1

    log.info("Downloading Kie asset → %s", path)
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with path.open("wb") as fp:
                    for chunk in response.iter_bytes(_CHUNK_SIZE):
                        if chunk:
                            fp.write(chunk)
    except httpx.HTTPError as exc:
        # Clean up a partial file.
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
        raise KieError(f"Failed to download {url}: {exc}") from exc

    return str(path)


def image_url_to_tensor(url: str) -> Any:
    """Download an image URL and return a ComfyUI-compatible tensor.

    ComfyUI expects images as ``torch.Tensor`` of shape ``(B, H, W, C)`` with
    float32 values in ``[0, 1]``. We import torch / numpy / PIL lazily.
    """
    import io

    import numpy as np
    import torch
    from PIL import Image

    raw = download_bytes(url)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    array = np.array(image, dtype=np.float32) / 255.0  # H, W, C
    tensor = torch.from_numpy(array).unsqueeze(0)  # 1, H, W, C
    return tensor


def first_url(data: dict[str, Any], keys: list[str]) -> str | None:
    """Pick the first non-empty URL from a Kie ``data`` dict.

    Different Kie endpoints return result URLs under different keys
    (``video_url``, ``image_url``, ``url``, ``output``, etc). This helper
    walks a list of candidate keys and returns the first that resolves to a
    non-empty string.
    """
    for key in keys:
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
        if isinstance(value, list) and value:
            for item in value:
                if isinstance(item, str) and item:
                    return item
                if isinstance(item, dict):
                    url = item.get("url") or item.get("image_url") or item.get("video_url")
                    if isinstance(url, str) and url:
                        return url
        if isinstance(value, dict):
            url = value.get("url") or value.get("image_url") or value.get("video_url")
            if isinstance(url, str) and url:
                return url
    return None
