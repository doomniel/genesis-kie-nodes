"""Helpers for downloading Kie.ai result URLs into local files / tensors.

ComfyUI nodes consume tensors for images and file paths for videos. This
module bridges the gap by:

- Downloading the URL bytes via httpx.
- For images: decoding to a torch tensor in the ComfyUI BHWC float32 format.
- For videos / audio: saving to ComfyUI's ``output/`` directory and returning
  the absolute path.

We keep torch / numpy / PIL imports lazy so this package can still be
imported in environments where ComfyUI is not present (e.g. tests).

Kie.ai result structure (after KieClient parses ``resultJson``):

    data = {
        "state": "success",
        "resultJson": '{"resultUrls": ["https://..."]}',  # raw JSON string
        "_parsed_result": {                                # added by KieClient
            "resultUrls": ["https://..."],
        },
        ...
    }

Use :func:`first_url` against ``data["_parsed_result"]`` (or against ``data``
directly — the helper looks in both places).

For DEDICATED endpoints (Veo, Runway, 4o, Flux Kontext), result URLs live in
``data.response.resultUrls`` (Veo, 4o) or ``data.videoInfo.videoUrl`` (Runway).
Use :func:`response_result_urls` for the Veo/4o shape.
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


def _safe_filename(url: str, prefix: str = "kie_output", fallback_ext: str = "bin") -> str:
    """Build a safe local filename from a URL."""
    try:
        parsed = urlparse(url)
        name = os.path.basename(parsed.path) or ""
    except Exception:  # noqa: BLE001
        name = ""
    if not name:
        name = f"{prefix}.{fallback_ext}"
    # Strip unsafe chars
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    if "." not in name:
        name = f"{name}.{fallback_ext}"
    # Avoid collisions
    out_dir = _comfyui_output_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / name
    counter = 1
    while target.exists():
        stem = target.stem
        suffix = target.suffix
        target = out_dir / f"{stem}_{counter}{suffix}"
        counter += 1
    return str(target)


def download_bytes(url: str) -> bytes:
    """Download a URL and return its raw bytes."""
    log.debug("Kie download bytes url=%s", url)
    try:
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content
    except httpx.HTTPError as exc:
        raise KieError(f"Failed to download {url}: {exc}") from exc


def download_to_output(
    url: str,
    *,
    prefix: str = "kie_output",
    fallback_ext: str = "bin",
) -> str:
    """Download a URL and save to ComfyUI's output dir. Returns absolute path."""
    target = _safe_filename(url, prefix=prefix, fallback_ext=fallback_ext)
    log.info("Kie downloading to %s", target)
    try:
        with httpx.Client(timeout=_DOWNLOAD_TIMEOUT_SECONDS, follow_redirects=True) as client:
            with client.stream("GET", url) as response:
                response.raise_for_status()
                with open(target, "wb") as fp:
                    for chunk in response.iter_bytes(_CHUNK_SIZE):
                        if chunk:
                            fp.write(chunk)
    except httpx.HTTPError as exc:
        raise KieError(f"Failed to download {url}: {exc}") from exc
    return target


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


def images_urls_to_tensor(urls: list[str]) -> Any:
    """Download multiple image URLs and stack as a single batch tensor.

    Output shape: ``(B, H, W, C)`` where B = len(urls).
    All images are resized to the smallest common (W, H) to allow stacking;
    if you need to preserve original sizes, call :func:`image_url_to_tensor`
    per URL instead.

    Used by endpoints that return multiple variants (e.g. 4o Image with
    nVariants > 1, Wan 2.7 Image with n > 1, Ideogram with num_images > 1).
    """
    import io

    import numpy as np
    import torch
    from PIL import Image

    if not urls:
        raise KieError("images_urls_to_tensor: no URLs provided.")

    # Load all images.
    pils: list[Image.Image] = []
    for url in urls:
        raw = download_bytes(url)
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        pils.append(img)

    # Single image fast path.
    if len(pils) == 1:
        array = np.array(pils[0], dtype=np.float32) / 255.0
        return torch.from_numpy(array).unsqueeze(0)

    # Find smallest common (W, H) so all fit into a single tensor.
    min_w = min(img.width for img in pils)
    min_h = min(img.height for img in pils)
    log.info(
        "images_urls_to_tensor: stacking %d images, resizing to (%d, %d)",
        len(pils), min_w, min_h,
    )

    arrays = []
    for img in pils:
        if (img.width, img.height) != (min_w, min_h):
            img = img.resize((min_w, min_h), Image.Resampling.LANCZOS)
        arrays.append(np.array(img, dtype=np.float32) / 255.0)

    stacked = np.stack(arrays, axis=0)  # B, H, W, C
    return torch.from_numpy(stacked)


def first_url(data: dict[str, Any], keys: list[str] | None = None) -> str | None:
    """Pick the first non-empty result URL from a Kie ``data`` dict.

    Kie stores the actual result inside ``resultJson`` (a JSON string).
    :class:`KieClient.wait_for_task` parses this into ``data["_parsed_result"]``
    automatically. This helper looks there first, then falls back to scanning
    legacy / direct keys for resilience.

    The standard Kie shape is::

        data["_parsed_result"] = {"resultUrls": ["https://...", ...]}

    Args:
        data: The ``data`` dict returned by ``wait_for_task`` or ``run``.
        keys: Optional list of additional fallback keys to scan after
            ``resultUrls`` (useful for non-standard responses or legacy
            endpoints).
    """
    # 1. Standard path: parsed resultJson with resultUrls list.
    parsed = data.get("_parsed_result") or {}
    if isinstance(parsed, dict):
        urls = parsed.get("resultUrls")
        if isinstance(urls, list) and urls:
            for item in urls:
                if isinstance(item, str) and item:
                    return item
                if isinstance(item, dict):
                    url = item.get("url") or item.get("imageUrl") or item.get("videoUrl")
                    if isinstance(url, str) and url:
                        return url

    # 2. Veo / 4o / Kontext dedicated shape: data.response.resultUrls
    response = data.get("response")
    if isinstance(response, dict):
        urls = response.get("resultUrls")
        if isinstance(urls, list) and urls:
            for item in urls:
                if isinstance(item, str) and item:
                    return item
        # Flux Kontext shape: data.response.resultImageUrl (singular string).
        # Note: response also has originImageUrl (input image), which we do NOT
        # surface — only the generated result.
        result_image = response.get("resultImageUrl")
        if isinstance(result_image, str) and result_image:
            return result_image

    # 3. Fallback: scan well-known direct keys on the data dict itself.
    fallback_keys = keys or [
        "video_url", "videoUrl", "video",
        "image_url", "imageUrl", "image",
        "audio_url", "audioUrl", "audio",
        "music_url", "url",
        "output_video_url", "output_image_url",
        "output", "result",
    ]
    for key in fallback_keys:
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


def all_urls(data: dict[str, Any]) -> list[str]:
    """Return ALL result URLs from a Kie ``data`` dict.

    Useful for endpoints that return multiple outputs (e.g. Nano Banana with
    ``num_images > 1``, Seedance with first/last frames, Wan 2.7 Image with
    ``n=4``, 4o Image with ``nVariants > 1``).

    Searches in priority order:
    1. ``data["_parsed_result"]["resultUrls"]``     (Market endpoints)
    2. ``data["response"]["resultUrls"]``           (Veo / 4o dedicated)
    """
    out: list[str] = []
    parsed = data.get("_parsed_result") or {}
    if isinstance(parsed, dict):
        urls = parsed.get("resultUrls")
        if isinstance(urls, list):
            for item in urls:
                if isinstance(item, str) and item:
                    out.append(item)
                elif isinstance(item, dict):
                    url = item.get("url") or item.get("imageUrl") or item.get("videoUrl")
                    if isinstance(url, str) and url:
                        out.append(url)
    if out:
        return out

    response = data.get("response")
    if isinstance(response, dict):
        urls = response.get("resultUrls")
        if isinstance(urls, list):
            for item in urls:
                if isinstance(item, str) and item:
                    out.append(item)
                elif isinstance(item, dict):
                    url = item.get("url") or item.get("imageUrl")
                    if isinstance(url, str) and url:
                        out.append(url)
        # Flux Kontext shape: data.response.resultImageUrl (singular string).
        # originImageUrl is the INPUT image — intentionally NOT surfaced.
        if not out:
            result_image = response.get("resultImageUrl")
            if isinstance(result_image, str) and result_image:
                out.append(result_image)
    return out
