"""API key + auth management for Kie.ai with GenesisLab proxy support.

This module handles two operating modes:

* **Direct mode** (default): ``KIE_BASE_URL`` is unset or points to
  ``api.kie.ai``. The Authorization header carries the real ``KIE_API_KEY``.

* **Proxy mode**: ``KIE_BASE_URL`` points to the GenesisLab proxy (e.g.
  ``https://app.genesislab.top/api/proxy/kie``). The Authorization header
  carries a synthetic token ``genesis-{GENESIS_USER_ID}`` which the proxy
  resolves to the real key (master or BYOK) and uses for wallet billing.

Mode is auto-detected from the base URL substring ``genesislab``. To force
proxy mode for testing, set ``GENESIS_PROXY_MODE=1``.
"""

from __future__ import annotations

import os
from pathlib import Path

from .exceptions import KieAuthError

_ENV_KEY = "KIE_API_KEY"
_DOTENV_FILENAMES = (".env", ".env.local")


def _load_dotenv_into_env() -> None:
    """Best-effort load of ``.env`` and ``.env.local`` from the current
    working directory and walk up to 4 levels. Does NOT overwrite values
    that are already present in ``os.environ``.

    We avoid pulling in python-dotenv as a dependency to keep this package
    minimal.
    """
    cwd = Path.cwd().resolve()
    candidates: list[Path] = []
    for parent in [cwd, *cwd.parents][:5]:
        for name in _DOTENV_FILENAMES:
            candidates.append(parent / name)

    for path in candidates:
        if not path.is_file():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            # Unreadable file — skip silently.
            continue


def get_api_key() -> str:
    """Return the configured raw Kie API key, raising :class:`KieAuthError`
    if none is available. Used in direct mode.
    """
    key = os.environ.get(_ENV_KEY)
    if not key:
        _load_dotenv_into_env()
        key = os.environ.get(_ENV_KEY)

    if not key:
        raise KieAuthError(
            f"Missing {_ENV_KEY}. Set it in your environment or in a .env "
            "file in the ComfyUI root before launching."
        )

    return key.strip()


def get_base_url() -> str:
    """Return the Kie.ai API base URL.

    Defaults to ``https://api.kie.ai`` but can be overridden via the
    ``KIE_BASE_URL`` environment variable (useful for routing through the
    GenesisLab proxy at e.g. ``https://app.genesislab.top/api/proxy/kie``).
    """
    return os.environ.get("KIE_BASE_URL", "https://api.kie.ai").rstrip("/")


def get_user_id() -> str | None:
    """Return the GenesisLab user id from the ``GENESIS_USER_ID`` env var.

    Required when running in proxy mode. The proxy uses this id to
    identify the wallet to charge and to authorize the call.
    """
    user_id = os.environ.get("GENESIS_USER_ID")
    if user_id:
        user_id = user_id.strip()
    return user_id or None


def is_proxied() -> bool:
    """Return True when ``KIE_BASE_URL`` points to the GenesisLab proxy.

    Detection is heuristic: any base URL containing 'genesislab' is treated
    as the proxy. This avoids forcing users to set a separate flag. To
    force proxy mode explicitly (e.g. for local testing against a tunnel),
    set ``GENESIS_PROXY_MODE=1``.
    """
    if os.environ.get("GENESIS_PROXY_MODE", "").strip() == "1":
        return True
    return "genesislab" in get_base_url().lower()


def get_auth_token() -> str:
    """Return the bearer token to use in the Authorization header.

    In proxy mode: returns ``genesis-{user_id}`` (the GenesisLab proxy
    resolves this to the real master/BYOK key and performs wallet billing).

    In direct mode: returns the raw ``KIE_API_KEY``.

    Raises :class:`KieAuthError` if the required configuration is missing
    for the detected mode.
    """
    if is_proxied():
        user_id = get_user_id()
        if not user_id:
            raise KieAuthError(
                "KIE_BASE_URL points to the GenesisLab proxy but "
                "GENESIS_USER_ID is not set. Configure it in your "
                "comfyui.service Environment= or in .env."
            )
        return f"genesis-{user_id}"
    return get_api_key()
