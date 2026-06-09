"""Upload tensors/blobs to the GenesisLab temp storage.

The ComfyUI nodes use this module to convert in-memory tensors
(IMAGE, AUDIO, VIDEO) into public URLs that external providers
(kie.ai, fal.ai, …) can download.

Backend endpoint: POST {backend_url}/api/upload-temp
   Headers:
     Authorization: Bearer genesis-{userId}
     Content-Type: image/png | audio/wav | video/mp4 | ...
   Body: raw bytes (max 50 MB)
   Response: {"url": "...", "key": "...", "expiresAt": "..."}

The returned URL points to a Cloudflare R2 bucket with a 24h lifecycle
rule, so we never accumulate storage. The bucket is publicly readable
so external providers can GET the URL without any auth.

The backend URL is derived from the existing KIE_BASE_URL env var
(by stripping the /api/proxy/kie suffix), so no new env config is
required when this module is colocated with the kie nodes.

Public API:
   upload_image_tensor(tensor)  -> str  (PNG, RGB or RGBA)
   upload_audio(audio_dict)     -> str  (WAV 16-bit PCM)
   upload_video_frames(frames)  -> str  (MP4 via imageio-ffmpeg)
   upload_bytes(body, content_type=...) -> str   (escape hatch)
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any

import requests

from .auth import get_user_id

log = logging.getLogger("genesis_kie.upload")


_DEFAULT_BACKEND = "https://app.genesislab.top"


def _get_backend_url() -> str:
    """Resolve the GenesisLab backend host.

    Priority:
      1. GENESIS_BACKEND_URL (explicit)
      2. Derive from KIE_BASE_URL (strip /api/proxy/kie suffix)
      3. Default to https://app.genesislab.top
    """
    explicit = os.environ.get("GENESIS_BACKEND_URL", "").strip().rstrip("/")
    if explicit:
        return explicit

    kie_base = os.environ.get("KIE_BASE_URL", "").strip().rstrip("/")
    if "/api/proxy/kie" in kie_base:
        return kie_base.split("/api/proxy/kie")[0].rstrip("/")

    return _DEFAULT_BACKEND


class UploadError(RuntimeError):
    """Raised when an upload to the GenesisLab backend fails."""


def upload_bytes(
    body: bytes,
    *,
    content_type: str,
    timeout: float = 60.0,
) -> str:
    """Upload raw bytes and return the public URL."""
    if not body:
        raise UploadError("empty body")

    user_id = get_user_id()
    if not user_id:
        raise UploadError(
            "GENESIS_USER_ID is not configured. Set it in the container "
            "environment or via FAL_KEY=genesis-{userId}."
        )

    backend = _get_backend_url()
    url = f"{backend}/api/upload-temp"
    headers = {
        "Authorization": f"Bearer genesis-{user_id}",
        "Content-Type": content_type,
    }

    log.info("[upload] POST %s (%d bytes, %s)", url, len(body), content_type)

    try:
        resp = requests.post(url, data=body, headers=headers, timeout=timeout)
    except requests.RequestException as exc:
        raise UploadError(f"network error: {exc}") from exc

    if resp.status_code >= 400:
        snippet = resp.text[:200] if resp.text else "(empty body)"
        raise UploadError(f"HTTP {resp.status_code}: {snippet}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise UploadError(f"invalid JSON response: {exc}") from exc

    public_url = data.get("url", "")
    if not public_url:
        raise UploadError(f"no 'url' in response: {data}")

    log.info("[upload] OK %s", public_url)
    return public_url


def upload_image_tensor(tensor: Any) -> str:
    """Encode a ComfyUI IMAGE tensor as PNG and upload it.

    ComfyUI IMAGE convention:
        torch.Tensor of shape (N, H, W, C) with float values in [0, 1].
    """
    import numpy as np
    from PIL import Image

    if tensor is None:
        raise UploadError("image tensor is None")

    if hasattr(tensor, "detach"):
        arr = tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(tensor)

    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim != 3:
        raise UploadError(
            f"expected image tensor of shape (N,H,W,C) or (H,W,C), got shape {arr.shape}"
        )

    channels = arr.shape[-1]
    if channels not in (1, 3, 4):
        raise UploadError(f"unsupported channel count: {channels}")

    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)

    if channels == 1:
        img = Image.fromarray(arr.squeeze(-1), mode="L").convert("RGB")
    elif channels == 3:
        img = Image.fromarray(arr, mode="RGB")
    else:
        img = Image.fromarray(arr, mode="RGBA")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return upload_bytes(buf.getvalue(), content_type="image/png")


def upload_audio(audio: dict) -> str:
    """Encode a ComfyUI AUDIO dict as WAV and upload it."""
    import numpy as np
    import scipy.io.wavfile as wavfile

    if not isinstance(audio, dict):
        raise UploadError(f"expected AUDIO dict, got {type(audio).__name__}")

    waveform = audio.get("waveform")
    sample_rate = audio.get("sample_rate")
    if waveform is None or sample_rate is None:
        raise UploadError("AUDIO dict missing 'waveform' or 'sample_rate' key")

    if hasattr(waveform, "detach"):
        arr = waveform.detach().cpu().numpy()
    else:
        arr = np.asarray(waveform)

    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise UploadError(
            f"expected audio of shape (B,C,T) or (C,T), got shape {arr.shape}"
        )

    if arr.shape[0] == 1:
        arr = arr[0]
    else:
        arr = arr.T

    arr = np.clip(arr, -1.0, 1.0)
    arr_int = (arr * 32767.0).astype(np.int16)

    buf = io.BytesIO()
    wavfile.write(buf, int(sample_rate), arr_int)
    return upload_bytes(buf.getvalue(), content_type="audio/wav")


def upload_video_frames(frames: Any, fps: int = 24) -> str:
    """Encode a ComfyUI IMAGE batch as MP4 (H.264) and upload it.

    Args:
        frames: ComfyUI IMAGE tensor of shape (N, H, W, C) with values in [0,1].
                A "video" in ComfyUI is just a batched image tensor.
        fps:    Output framerate (default 24).

    Returns:
        Public URL of the uploaded MP4 (lives in R2 temp storage).
    """
    import numpy as np
    import imageio

    if frames is None:
        raise UploadError("frames tensor is required")
    if not hasattr(frames, "shape") or len(frames.shape) != 4:
        shape = getattr(frames, "shape", None)
        raise UploadError(
            f"expected IMAGE tensor of shape (N,H,W,C), got shape {shape}"
        )

    # Tensor → uint8 numpy
    if hasattr(frames, "detach"):
        arr = frames.detach().cpu().numpy()
    else:
        arr = np.asarray(frames)
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    n, h, w, c = arr.shape
    if c == 1:
        # Grayscale → expand to RGB
        arr = np.repeat(arr, 3, axis=-1)
    elif c == 4:
        # RGBA → drop alpha (MP4 doesn't support alpha cleanly)
        arr = arr[..., :3]

    # H.264 requires even dimensions
    if h % 2 != 0 or w % 2 != 0:
        new_h = h - (h % 2)
        new_w = w - (w % 2)
        arr = arr[:, :new_h, :new_w, :]

    # FFMPEG plugin in imageio requires a real path (it spawns a subprocess
    # that can't write to a BytesIO). Encode to a tempfile, then read+upload.
    import tempfile
    import os as _os

    fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
    _os.close(fd)
    try:
        with imageio.get_writer(
            tmp_path,
            format="ffmpeg",
            fps=fps,
            codec="libx264",
            pixelformat="yuv420p",
            quality=8,
            macro_block_size=1,
        ) as writer:
            for frame in arr:
                writer.append_data(frame)
        with open(tmp_path, "rb") as f:
            mp4_bytes = f.read()
    finally:
        if _os.path.exists(tmp_path):
            _os.unlink(tmp_path)

    return upload_bytes(mp4_bytes, content_type="video/mp4")
