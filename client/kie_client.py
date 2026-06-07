"""HTTP client for Kie.ai.

Kie.ai exposes FIVE different endpoint patterns:

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

3. **Runway dedicated** (Runway Gen-4 + Aleph):

   - POST /api/v1/runway/generate  with { prompt, imageUrl, quality, ... }
     POST /api/v1/runway/extend    with { taskId, prompt, quality, ... }
     POST /api/v1/aleph/generate   with { prompt, videoUrl, waterMark, ... }
     (parameters use camelCase and are NOT wrapped in an ``input`` key)
   - GET  /api/v1/runway/record-detail?taskId=X
   - Status lives in ``data.state`` (wait/success/fail)
   - Result URL lives in ``data.videoInfo.videoUrl``

4. **4o Image dedicated** (OpenAI GPT 4o Image):

   - POST /api/v1/gpt4o-image/generate with { prompt, filesUrl[], size, ... }
     (camelCase at top level, no ``input`` wrapper)
   - GET  /api/v1/gpt4o-image/record-info?taskId=X
   - Status lives in ``data.successFlag`` (Veo-style: 0/1/2/3)
   - Result URLs live in ``data.response.resultUrls`` (array)

5. **Flux Kontext dedicated** (BFL Flux Kontext):

   - POST /api/v1/flux/kontext/generate with { prompt, model, aspectRatio,
     enableTranslation, outputFormat, promptUpsampling, safetyTolerance, ... }
   - GET  /api/v1/flux/kontext/record-info?taskId=X
   - Status lives in ``data.successFlag`` (Veo-style: 0/1/2/3)
   - Result URLs live in ``data.response.resultUrls``

This client exposes method pairs per pattern, plus convenience
``run_market`` / ``run_veo`` / ``run_runway`` / ``run_4o_image`` /
``run_flux_kontext`` wrappers that hide the polling details.

For the Gemini Omni helper endpoints (audio/character creation), see
``create_omni_resource`` — these are synchronous and return immediately.
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

# Runway endpoint state values (text strings, like Market).
_RUNWAY_STATE_COMPLETED = {"success"}
_RUNWAY_STATE_FAILED = {"fail"}

# Inner ``code`` values that mean SUCCESS at the API level.
# Note: omni/audio/create uses 0 for success (not 200).
_OK_CODES = {0, 200}

# Polling defaults.
_DEFAULT_POLL_INTERVAL_SECONDS = 3.0
_DEFAULT_TIMEOUT_SECONDS = 600.0
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


class KieClient:
    """Thin HTTP client around the Kie.ai REST API.

    Supports Market generic, Veo dedicated, Runway dedicated, and Omni
    resource-creation endpoints.
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
                "User-Agent": "comfyui-genesis-kie/0.3.0",
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
        """Poll a market task until success/fail/timeout."""
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
                    "Kie market task succeeded taskId=%s attempts=%d",
                    task_id, attempts,
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
        """Create a Veo 3.1 task using the dedicated API."""
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

    # ============================================== RUNWAY PATTERN (dedicated)

    def create_runway_task(
        self,
        path: str,
        body: dict[str, Any],
    ) -> str:
        """Create a Runway-family task.

        Args:
            path: One of "/api/v1/runway/generate", "/api/v1/runway/extend",
                  "/api/v1/aleph/generate".
            body: Pre-built request body (camelCase keys, no ``input`` wrapper).

        Note: Runway endpoints use camelCase keys directly at the top
        level (imageUrl, aspectRatio, waterMark, callBackUrl) — NOT
        wrapped in an ``input`` key like Market endpoints.
        """
        log.debug("Kie createRunwayTask path=%s", path)
        payload = self._request("POST", path, json=body)
        data = payload.get("data") or {}
        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise KieError(
                f"Kie.ai response did not include a taskId: {payload}"
            )
        log.info("Kie runway task created path=%s taskId=%s", path, task_id)
        return task_id

    def get_runway_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a Runway task's current state.

        Uses the same record-detail endpoint regardless of whether the task
        was generate/extend/aleph.
        """
        payload = self._request(
            "GET",
            "/api/v1/runway/record-detail",
            params={"taskId": task_id},
        )
        return payload.get("data") or {}

    def wait_for_runway_task(
        self,
        task_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Poll a Runway task until completion.

        State values: "wait" / "success" / "fail".
        Result URL: ``data.videoInfo.videoUrl``.
        """
        deadline = time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            data = self.get_runway_task(task_id)
            state = (data.get("state") or "").lower()

            if progress_callback is not None:
                try:
                    progress_callback(data)
                except Exception:  # noqa: BLE001
                    log.debug("Progress callback raised; ignoring", exc_info=True)

            if state in _RUNWAY_STATE_COMPLETED:
                log.info(
                    "Kie runway task succeeded taskId=%s attempts=%d",
                    task_id, attempts,
                )
                return data

            if state in _RUNWAY_STATE_FAILED:
                err_code = data.get("failCode") or ""
                err_msg = (
                    data.get("failMsg")
                    or data.get("msg")
                    or f"Runway task failed (state={state})"
                )
                err = f"{err_msg} (failCode={err_code})" if err_code else err_msg
                raise KieTaskFailedError(err, task_id=task_id)

            if time.monotonic() >= deadline:
                raise KieTimeoutError(
                    f"Runway task {task_id} did not complete within {timeout:.0f}s "
                    f"(last state={state!r})"
                )

            time.sleep(poll_interval)

    def run_runway(
        self,
        path: str,
        body: dict[str, Any],
        *,
        poll_interval: float = 5.0,
        timeout: float = 900.0,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Convenience: create + poll a Runway task."""
        task_id = self.create_runway_task(path, body)
        return self.wait_for_runway_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    # =============================================== OMNI RESOURCE ENDPOINTS

    def create_omni_audio(
        self,
        *,
        audio_id: str,
        name: str,
        voice_description: str,
        example_dialogue: str = "",
    ) -> dict[str, Any]:
        """Create a Gemini Omni reusable voice (synchronous).

        Returns the response ``data`` dict, including ``kieAudioId`` to
        reference later in Gemini Omni Video calls.
        """
        body = {
            "audio_id": audio_id,
            "name": name,
            "voice_description": voice_description,
        }
        if example_dialogue:
            body["example_dialogue"] = example_dialogue

        log.debug("Kie create omni audio audio_id=%s", audio_id)
        payload = self._request("POST", "/api/v1/omni/audio/create", json=body)
        return payload.get("data") or {}

    def create_omni_character(
        self,
        *,
        character_id: str,
        name: str,
        character_description: str,
        image_urls: list[str],
    ) -> dict[str, Any]:
        """Create a Gemini Omni reusable character (synchronous).

        Returns the response ``data`` dict, including the kieCharacterId
        to reference later in Gemini Omni Video calls.

        Note: The exact endpoint shape was inferred from the audio
        endpoint pattern. If Kie's character endpoint differs, adjust
        the path or body shape here.
        """
        body = {
            "character_id": character_id,
            "name": name,
            "character_description": character_description,
            "image_urls": image_urls,
        }
        log.debug("Kie create omni character character_id=%s", character_id)
        payload = self._request(
            "POST",
            "/api/v1/omni/character/create",
            json=body,
        )
        return payload.get("data") or {}

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

    # =============================================== 4O IMAGE PATTERN (dedicated)

    def create_4o_image_task(
        self,
        *,
        prompt: str,
        files_url: list[str] | None = None,
        size: str = "1:1",
        n_variants: int = 1,
        is_enhance: bool = False,
        upload_cn: bool = False,
        enable_fallback: bool = False,
        fallback_model: str | None = None,
        callback_url: str | None = None,
    ) -> str:
        """Create a 4o Image task using the dedicated API.

        Endpoint: ``POST /api/v1/gpt4o-image/generate``

        Body uses camelCase top-level keys (filesUrl, callBackUrl, isEnhance,
        uploadCn, enableFallback, fallbackModel) — NOT wrapped in ``input``.

        Note: nVariants is informally documented; if rejected, drop it and
        call the endpoint multiple times.
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "size": size,
            "isEnhance": is_enhance,
            "uploadCn": upload_cn,
            "enableFallback": enable_fallback,
            "nVariants": n_variants,
        }
        if files_url:
            body["filesUrl"] = files_url
        if fallback_model:
            body["fallbackModel"] = fallback_model
        if callback_url:
            body["callBackUrl"] = callback_url

        log.debug("Kie create4oImageTask")
        payload = self._request("POST", "/api/v1/gpt4o-image/generate", json=body)
        data = payload.get("data") or {}
        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise KieError(f"Kie.ai response did not include a taskId: {payload}")
        log.info("Kie 4o image task created taskId=%s", task_id)
        return task_id

    def get_4o_image_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a 4o image task's current state."""
        payload = self._request(
            "GET",
            "/api/v1/gpt4o-image/record-info",
            params={"taskId": task_id},
        )
        return payload.get("data") or {}

    def wait_for_4o_image_task(
        self,
        task_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Poll a 4o image task until completion.

        Uses successFlag like Veo:
        - 0 = generating
        - 1 = success
        - 2 = failed
        - 3 = generation failed
        """
        deadline = time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            data = self.get_4o_image_task(task_id)
            flag = data.get("successFlag")

            if progress_callback is not None:
                try:
                    progress_callback(data)
                except Exception:  # noqa: BLE001
                    log.debug("Progress callback raised; ignoring", exc_info=True)

            if flag == _VEO_FLAG_SUCCESS:
                log.info(
                    "Kie 4o image task succeeded taskId=%s attempts=%d",
                    task_id, attempts,
                )
                return data

            if flag in (_VEO_FLAG_FAILED, _VEO_FLAG_GEN_FAILED):
                err_code = data.get("errorCode")
                err_msg = (
                    data.get("errorMessage")
                    or data.get("msg")
                    or f"4o image task failed (flag={flag})"
                )
                err = f"{err_msg} (errorCode={err_code})" if err_code else err_msg
                raise KieTaskFailedError(err, task_id=task_id)

            if time.monotonic() >= deadline:
                raise KieTimeoutError(
                    f"4o image task {task_id} did not complete within {timeout:.0f}s "
                    f"(last successFlag={flag})"
                )

            time.sleep(poll_interval)

    def run_4o_image(
        self,
        *,
        prompt: str,
        files_url: list[str] | None = None,
        size: str = "1:1",
        n_variants: int = 1,
        is_enhance: bool = False,
        upload_cn: bool = False,
        enable_fallback: bool = False,
        fallback_model: str | None = None,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Convenience: create + poll a 4o image task."""
        task_id = self.create_4o_image_task(
            prompt=prompt,
            files_url=files_url,
            size=size,
            n_variants=n_variants,
            is_enhance=is_enhance,
            upload_cn=upload_cn,
            enable_fallback=enable_fallback,
            fallback_model=fallback_model,
        )
        return self.wait_for_4o_image_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    # ========================================== FLUX KONTEXT PATTERN (dedicated)

    def create_flux_kontext_task(
        self,
        *,
        prompt: str,
        model: str = "flux-kontext-pro",
        input_image: str | None = None,
        aspect_ratio: str | None = None,
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        enable_translation: bool = True,
        safety_tolerance: int = 2,
        callback_url: str | None = None,
    ) -> str:
        """Create a Flux Kontext task using the dedicated API.

        Endpoint: ``POST /api/v1/flux/kontext/generate``

        Models: ``flux-kontext-pro`` or ``flux-kontext-max``.
        If ``input_image`` is provided, becomes an image-edit task; otherwise
        text-to-image.
        """
        body: dict[str, Any] = {
            "prompt": prompt,
            "model": model,
            "outputFormat": output_format,
            "promptUpsampling": prompt_upsampling,
            "enableTranslation": enable_translation,
            "safetyTolerance": safety_tolerance,
        }
        if input_image:
            body["inputImage"] = input_image
        if aspect_ratio:
            body["aspectRatio"] = aspect_ratio
        if callback_url:
            body["callBackUrl"] = callback_url

        log.debug("Kie createFluxKontextTask model=%s", model)
        payload = self._request("POST", "/api/v1/flux/kontext/generate", json=body)
        data = payload.get("data") or {}
        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise KieError(f"Kie.ai response did not include a taskId: {payload}")
        log.info("Kie flux kontext task created model=%s taskId=%s", model, task_id)
        return task_id

    def get_flux_kontext_task(self, task_id: str) -> dict[str, Any]:
        """Fetch a Flux Kontext task's current state."""
        payload = self._request(
            "GET",
            "/api/v1/flux/kontext/record-info",
            params={"taskId": task_id},
        )
        return payload.get("data") or {}

    def wait_for_flux_kontext_task(
        self,
        task_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Poll a Flux Kontext task until completion.

        Uses successFlag like Veo/4o (0=gen, 1=success, 2/3=failed).
        """
        deadline = time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            data = self.get_flux_kontext_task(task_id)
            flag = data.get("successFlag")

            if progress_callback is not None:
                try:
                    progress_callback(data)
                except Exception:  # noqa: BLE001
                    log.debug("Progress callback raised; ignoring", exc_info=True)

            if flag == _VEO_FLAG_SUCCESS:
                log.info(
                    "Kie flux kontext task succeeded taskId=%s attempts=%d",
                    task_id, attempts,
                )
                return data

            if flag in (_VEO_FLAG_FAILED, _VEO_FLAG_GEN_FAILED):
                err_code = data.get("errorCode")
                err_msg = (
                    data.get("errorMessage")
                    or data.get("msg")
                    or f"Flux kontext task failed (flag={flag})"
                )
                err = f"{err_msg} (errorCode={err_code})" if err_code else err_msg
                raise KieTaskFailedError(err, task_id=task_id)

            if time.monotonic() >= deadline:
                raise KieTimeoutError(
                    f"Flux kontext task {task_id} did not complete within {timeout:.0f}s "
                    f"(last successFlag={flag})"
                )

            time.sleep(poll_interval)

    def run_flux_kontext(
        self,
        *,
        prompt: str,
        model: str = "flux-kontext-pro",
        input_image: str | None = None,
        aspect_ratio: str | None = None,
        output_format: str = "jpeg",
        prompt_upsampling: bool = False,
        enable_translation: bool = True,
        safety_tolerance: int = 2,
        poll_interval: float = 3.0,
        timeout: float = 600.0,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Convenience: create + poll a Flux Kontext task."""
        task_id = self.create_flux_kontext_task(
            prompt=prompt,
            model=model,
            input_image=input_image,
            aspect_ratio=aspect_ratio,
            output_format=output_format,
            prompt_upsampling=prompt_upsampling,
            enable_translation=enable_translation,
            safety_tolerance=safety_tolerance,
        )
        return self.wait_for_flux_kontext_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

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
        try:
            return int(data) if data is not None else None
        except (TypeError, ValueError):
            return None
