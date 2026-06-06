"""HTTP client for Kie.ai.

Kie.ai uses an async task-based model:

1. POST /api/v1/jobs/createTask  with { model, input }
   → returns { code, msg, data: { taskId } }

2. GET /api/v1/jobs/recordInfo?taskId={taskId}  (polling)
   → returns { code, msg, data: { status, output, ... } }

All nodes in this package delegate to :class:`KieClient` for the actual HTTP
work. Individual nodes only declare their model name and input parameters.
"""

from __future__ import annotations

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

# Status values returned by Kie.ai. We treat anything not in COMPLETED/FAILED
# as "still running".
_STATUS_COMPLETED = {"completed", "success", "succeeded", "finished"}
_STATUS_FAILED = {"failed", "error", "cancelled", "canceled"}

# Polling defaults. These can be overridden per-call.
_DEFAULT_POLL_INTERVAL_SECONDS = 3.0
_DEFAULT_TIMEOUT_SECONDS = 600.0  # 10 minutes
_DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0


class KieClient:
    """Thin HTTP client around the Kie.ai REST API.

    The client is intentionally stateless: each ComfyUI node that needs Kie
    can instantiate one on the fly. Authentication and base URL are resolved
    at construction time from environment variables.
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

        if response.status_code == 401 or response.status_code == 403:
            raise KieAuthError(
                f"Authentication failed ({response.status_code}). "
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
                f"Kie.ai server error ({response.status_code}): {response.text}",
                code=response.status_code,
            )

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise KieError(
                f"Invalid JSON response from {path}: {response.text[:200]}"
            ) from exc

        # Kie wraps everything in { code, msg, data }. A non-200 inner code is
        # treated as an error even if HTTP says 200.
        inner_code = payload.get("code")
        if inner_code is not None and inner_code != 200:
            raise KieError(
                payload.get("msg") or f"Unknown error (code={inner_code})",
                code=inner_code,
            )

        return payload

    # ------------------------------------------------------- task lifecycle

    def create_task(self, model: str, inputs: dict[str, Any],
                    callback_url: str | None = None) -> str:
        """Create a new async generation task.

        Args:
            model: The Kie.ai model identifier (e.g. ``"google/veo-3.1-fast"``).
            inputs: Model-specific input dict.
            callback_url: Optional webhook URL that Kie.ai will POST to on
                completion. If provided, polling is not strictly required.

        Returns:
            The ``taskId`` assigned by Kie.ai.
        """
        body: dict[str, Any] = {
            "model": model,
            "input": inputs,
        }
        if callback_url:
            body["callBackUrl"] = callback_url

        log.debug("Kie createTask model=%s", model)
        payload = self._request("POST", "/api/v1/jobs/createTask", json=body)

        data = payload.get("data") or {}
        task_id = data.get("taskId") or data.get("task_id")
        if not task_id:
            raise KieError(
                "Kie.ai response did not include a taskId: "
                f"{payload}"
            )
        log.info("Kie task created model=%s taskId=%s", model, task_id)
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any]:
        """Fetch the current state of a task. Returns the ``data`` portion
        of the response.
        """
        payload = self._request(
            "GET",
            "/api/v1/jobs/recordInfo",
            params={"taskId": task_id},
        )
        return payload.get("data") or {}

    def wait_for_task(
        self,
        task_id: str,
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Block until the task completes, fails, or times out.

        Args:
            task_id: Task identifier returned by :meth:`create_task`.
            poll_interval: Seconds between polling requests.
            timeout: Maximum total time to wait, in seconds.
            progress_callback: Optional callable that receives each polled
                ``data`` dict — useful for surfacing progress in ComfyUI.

        Returns:
            The final ``data`` dict from Kie.ai.
        """
        deadline = time.monotonic() + timeout
        attempts = 0

        while True:
            attempts += 1
            data = self.get_task(task_id)
            status = (data.get("status") or "").lower()

            if progress_callback is not None:
                try:
                    progress_callback(data)
                except Exception:  # noqa: BLE001
                    log.debug("Progress callback raised; ignoring", exc_info=True)

            if status in _STATUS_COMPLETED:
                log.info(
                    "Kie task completed taskId=%s attempts=%d",
                    task_id,
                    attempts,
                )
                return data

            if status in _STATUS_FAILED:
                err_msg = (
                    data.get("errorMessage")
                    or data.get("error")
                    or data.get("msg")
                    or "Task failed without a specific error message."
                )
                raise KieTaskFailedError(err_msg, task_id=task_id)

            if time.monotonic() >= deadline:
                raise KieTimeoutError(
                    f"Task {task_id} did not complete within {timeout:.0f}s "
                    f"(last status={status!r})"
                )

            time.sleep(poll_interval)

    # ---------------------------------------------------- one-shot convenience

    def run(
        self,
        model: str,
        inputs: dict[str, Any],
        *,
        poll_interval: float = _DEFAULT_POLL_INTERVAL_SECONDS,
        timeout: float = _DEFAULT_TIMEOUT_SECONDS,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Create a task and block until it finishes.

        This is the method most node implementations should use.
        """
        task_id = self.create_task(model, inputs)
        return self.wait_for_task(
            task_id,
            poll_interval=poll_interval,
            timeout=timeout,
            progress_callback=progress_callback,
        )

    # ---------------------------------------------------------- account utils

    def get_credits_remaining(self) -> int | None:
        """Best-effort fetch of remaining credits. Returns ``None`` if the
        endpoint is unavailable.
        """
        try:
            payload = self._request("GET", "/api/v1/chat/credit")
        except KieError:
            return None
        data = payload.get("data") or {}
        credits = data.get("credits") or data.get("balance")
        try:
            return int(credits) if credits is not None else None
        except (TypeError, ValueError):
            return None
