"""HTTP client for Kie.ai.

Kie.ai exposes TWO different endpoint patterns:

1. **Market generic** (Seedance, Kling, Hailuo, Wan, ElevenLabs, Claude, etc):

   - POST /api/v1/jobs/createTask  with { model, input: {...} }
   - GET  /api/v1/jobs/recordInfo?taskId=X
   - Status lives in ``data.state`` (waiting/queuing/generating/success/fail)
   - Result URLs live in ``data.resultJson`` (a JSON-encoded STRING)
     containing ``{"resultUrls": [...]}`` or ``{"resultObject": {...}}``.

2. **Veo dedicated** (Veo 3.1 only):

   - POST /api/v1/veo/generate     with { prompt, model: "veo3_fast", ... }
     (parameters are NOT wrapped in an ``input`` key)
   - GET  /api/v1/veo/record-info?taskId=X
   - Status lives in ``data.successFlag`` (0=generating, 1=success, 2/3=failed)
   - Result URLs live in ``data.response.resultUrls`` (array, NOT a JSON string)

This client exposes two pairs of methods, one per pattern, plus convenience
``run_market`` / ``run_veo`` wrappers that hide the polling details.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import httpx

from .auth import get_api_key, get_base_url
from .exceptions import (
    KieAuthError,
    KieError,
    KieRateLimitError,
    KieTaskFailedError,
    KieTimeoutError,
)

log = logging.getLogger("genesis_kie")

# Market endpoint state values.
_MARKET_STATE_COMPLETED = {"success"}
_MARKET_STATE_FAILED = {"fail"}

# Veo endpoint successFlag values.
_VEO_FLAG_GENERATING = 0
_VEO_FLAG_SUCCESS = 1
_VEO_FLAG_FAILED = 2
_VEO_FLAG_GEN_FAILED = 3

# Inner ``code`` values that mean SUCCESS at the API level.
_OK_CODES = {200}

# Polling defaults.
_DEFAULT_POLL_INTERVAL_SECONDS = 3.0
_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


class KieClient:
    """Thin HTTP client around the Kie.ai REST API.

    Supports both the Market generic endpoints and the Veo dedicated endpoints.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        request_timeout: float = _DEFAULT_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        self._api_key = api_key or get_api_key()
        self._base_url = (base_url or get_base_url()).rstrip("/")
        self._client = httpx.Client(
            timeout=httpx.Timeout(request_timeout),
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "User-Agent": "comfyui-genesis-kie/0.1.0",
            },
        )

    # ------------------------------------------------------------------ utils

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "KieClient":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()

    # ----------------------------------------------------------- low-level IO

    def _request(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        url = f"{self._base_url}{path}"
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            raise KieTimeoutError(
                f"HTTP {method} {path} timed out: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise KieError(f"HTTP {method} {path} failed: {exc}") from exc

        if response.status_code in (401, 403):
            raise KieAuthError(
                f"Authentication failed (HTTP {response.status_code}). "
                "Verify your KIE_API_KEY.",
                code=response.status_code,
            )
        if response.status_code == 429:
            raise KieRateLimitError(
                "Rate limit exceeded (HTTP 429). Slow down requests or "
                "contact Kie.ai support for a higher limit.",
                code=429,
            )
        if response.status_code >= 500:
            raise KieError(
                f"Kie.ai server error (HTTP {response.status_code}): "
                f"{response.text[:200]}",
                code=response.status_code,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise KieError(
                f"Invalid JSON response from {path}: {response.text[:200]}"
            ) from exc

        inner_code = payload.get("code")
        if inner_code is not None and inner_code not in _OK_CODES:
            msg = payload.get("msg") or f"Unknown error (code={inner_code})"
            if inner_code in (401, 403):
                raise KieAuthError(msg, code=inner_code)
            if inner_code == 429:
                raise KieRateLimitError(msg, code=inner_code)
            raise KieError(msg, code=inner_code)

        return payload

    # =============================================== MARKET PATTERN (generic)

    def create_market_task(
        self,
        model: str,
        inputs: dict[str, Any],
        callback_url: str | None = None,
    ) -> str:
        """Create a task in the Market generic endpoint.

        Used for: Seedance, Kling, Hailuo, Wan, HappyHorse, Grok Imagine,
        ElevenLabs, Claude, GPT, Gemini, Nano Banana, Ideogram, etc.
        """
        body: dict[str, Any] = {"model": model, "input": inputs}
        if callback_url:
            body["callBackUrl"] = callback_url

        log.debug("Kie createMarketTask model=%s", model)
        payload = self._request("POST", "/api/v1/jobs/createTask", json=body)

        data = payload.get("data") or {}
        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise KieError(
                f"Kie.ai response did not include a taskId: {payload}"
            )
        log.info("Kie market task created model=%s taskId=%s", model, task_id)
        return task_id

    def get_market_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a market task's current state."""
        payload = self._request(
            "GET",
            "/api/v1/jobs/recordInfo",
            params={"taskId": task_id},
        )
        return payload.get("data") or {}

    def wait_for_market_task(
        self,
        task_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Poll a market task until success/fail/timeout.

        Returns the ``data`` dict with ``_parsed_result`` (the parsed
        ``resultJson``) merged in.
        """
        deadline = time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            data = self.get_market_task(task_id)
            state = (data.get("state") or "").lower()

            if progress_callback is not None:
                try:
                    progress_callback(data)
                except Exception:  # noqa: BLE001
                    log.debug("Progress callback raised; ignoring", exc_info=True)

            if state in _MARKET_STATE_COMPLETED:
                data["_parsed_result"] = self._parse_result_json(data)
                log.info(
                    "Kie market task succeeded taskId=%s attempts=%d cost_time=%sms",
                    task_id, attempts, data.get("costTime"),
                )
                return data

            if state in _MARKET_STATE_FAILED:
                fail_code = data.get("failCode") or ""
                fail_msg = (
                    data.get("failMsg")
                    or data.get("msg")
                    or "Task failed without a specific error message."
                )
                err = f"{fail_msg} (failCode={fail_code})" if fail_code else fail_msg
                raise KieTaskFailedError(err, task_id=task_id)

            if time.monotonic() >= deadline:
                raise KieTimeoutError(
                    f"Market task {task_id} did not complete within {timeout:.0f}s "
                    f"(last state={state!r})"
                )

            time.sleep(poll_interval)

    def run_market(
        self,
        model: str,
        inputs: dict[str, Any],
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Convenience: create + poll a market task."""
        task_id = self.create_market_task(model, inputs)
        return self.wait_for_market_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    # ================================================= VEO PATTERN (dedicated)

    def create_veo_task(
        self,
        *,
        prompt: str,
        model: str = "veo3_fast",
        image_urls: list[str] | None = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        generation_type: str | None = None,
        enable_translation: bool = True,
        watermark: str | None = None,
        callback_url: str | None = None,
        seeds: int | None = None,
    ) -> str:
        """Create a Veo 3.1 task using the dedicated API.

        Args:
            prompt: Required text prompt.
            model: One of ``"veo3"``, ``"veo3_fast"``, ``"veo3_lite"``.
            image_urls: Optional list of 1-2 image URLs (for image-to-video).
            aspect_ratio: ``"16:9"``, ``"9:16"``, or ``"Auto"``.
            resolution: ``"720p"``, ``"1080p"``, or ``"4k"``.
            generation_type: ``"TEXT_2_VIDEO"``, ``"FIRST_AND_LAST_FRAMES_2_VIDEO"``,
                or ``"REFERENCE_2_VIDEO"``. Auto-detected if omitted.
            enable_translation: Translate prompt to English (default True).
            watermark: Optional watermark text.
            callback_url: Optional webhook URL.
            seeds: Optional seed for reproducibility.
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "aspect_ratio": aspect_ratio,
            "resolution": resolution,
            "enableTranslation": enable_translation,
        }
        if image_urls:
            body["imageUrls"] = image_urls
        if generation_type:
            body["generationType"] = generation_type
        if watermark:
            body["watermark"] = watermark
        if callback_url:
            body["callBackUrl"] = callback_url
        if seeds is not None:
            body["seeds"] = seeds

        log.debug("Kie createVeoTask model=%s", model)
        payload = self._request("POST", "/api/v1/veo/generate", json=body)

        data = payload.get("data") or {}
        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise KieError(
                f"Kie.ai response did not include a taskId: {payload}"
            )
        log.info("Kie veo task created model=%s taskId=%s", model, task_id)
        return task_id

    def get_veo_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a Veo task's current state."""
        payload = self._request(
            "GET",
            "/api/v1/veo/record-info",
            params={"taskId": task_id},
        )
        return payload.get("data") or {}

    def wait_for_veo_task(
        self,
        task_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Poll a Veo task until completion."""
        deadline = time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            data = self.get_veo_task(task_id)
            flag = data.get("successFlag")

            if progress_callback is not None:
                try:
                    progress_callback(data)
                except Exception:  # noqa: BLE001
                    log.debug("Progress callback raised; ignoring", exc_info=True)

            if flag == _VEO_FLAG_SUCCESS:
                log.info(
                    "Kie veo task succeeded taskId=%s attempts=%d",
                    task_id, attempts,
                )
                return data

            if flag in (_VEO_FLAG_FAILED, _VEO_FLAG_GEN_FAILED):
                err_code = data.get("errorCode")
                err_msg = (
                    data.get("errorMessage")
                    or data.get("msg")
                    or f"Veo task failed (flag={flag})"
                )
                err = f"{err_msg} (errorCode={err_code})" if err_code else err_msg
                raise KieTaskFailedError(err, task_id=task_id)

            if time.monotonic() >= deadline:
                raise KieTimeoutError(
                    f"Veo task {task_id} did not complete within {timeout:.0f}s "
                    f"(last successFlag={flag})"
                )

            time.sleep(poll_interval)

    def run_veo(
        self,
        *,
        prompt: str,
        model: str = "veo3_fast",
        image_urls: list[str] | None = None,
        aspect_ratio: str = "16:9",
        resolution: str = "720p",
        generation_type: str | None = None,
        enable_translation: bool = True,
        watermark: str | None = None,
        seeds: int | None = None,
        poll_interval: float = 5.0,
        timeout: float = 900.0,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Convenience: create + poll a Veo task."""
        task_id = self.create_veo_task(
            prompt=prompt,
            model=model,
            image_urls=image_urls,
            aspect_ratio=aspect_ratio,
            resolution=resolution,
            generation_type=generation_type,
            enable_translation=enable_translation,
            watermark=watermark,
            seeds=seeds,
        )
        return self.wait_for_veo_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    # ------------------------------------------------------------- helpers

    @staticmethod
    def _parse_result_json(data: dict[str, Any]) -> dict[str, Any]:
        """Parse the ``resultJson`` string into a dict (Market endpoints only)."""
        raw = data.get("resultJson")
        if not raw:
            return {}
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (ValueError, TypeError):
                log.warning("Could not parse resultJson: %r", raw[:200])
                return {}
        return {}

    # ---------------------------------------------------------- account utils

    def get_credits_remaining(self) -> int | None:
        """Best-effort fetch of remaining credits."""
        try:
            payload = self._request("GET", "/api/v1/chat/credit")
        except KieError:
            return None
        data = payload.get("data")
        if isinstance(data, dict):
            credits = (
                data.get("credits")
                or data.get("balance")
                or data.get("remaining")
            )
            try:
                return int(credits) if credits is not None else None
            except (TypeError, ValueError):
                return None
        # Sometimes the endpoint returns the number directly
        try:
            return int(data) if data is not None else None
        except (TypeError, ValueError):
            return None
