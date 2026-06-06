"""API key management for Kie.ai.

The key is read from the environment variable ``KIE_API_KEY``. When running
inside GenesisLab, this variable is injected automatically by the proxy at
container startup. For standalone use, set it manually before launching
ComfyUI.
"""

from __future__ import annotations

import os
from pathlib import Path

from .exceptions import KieAuthError

_ENV_KEY = "KIE_API_KEY"
_DOTENV_FILENAMES = (".env", ".env.local")


def _load_dotenv_into_env() -> None:
    """Best-effort load of ``.env`` and ``.env.local`` from the current working
    directory and walk up to 4 levels. Does NOT overwrite values that are
    already present in ``os.environ``.

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
    """Return the configured Kie API key, raising :class:`KieAuthError` if
    none is available.
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
    GenesisLab proxy at e.g. ``https://kie.genesislab.top``).
    """
    return os.environ.get("KIE_BASE_URL", "https://api.kie.ai").rstrip("/")
