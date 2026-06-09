"""HTTP client utilities for talking to Kie.ai."""

from .auth import get_api_key, get_base_url, get_auth_token, get_user_id, is_proxied
from .download import (
    all_urls,
    download_bytes,
    download_to_output,
    first_url,
    image_url_to_tensor,
    images_urls_to_tensor,
)
from .exceptions import (
    KieAuthError,
    KieError,
    KieRateLimitError,
    KieTaskFailedError,
    KieTimeoutError,
    KieValidationError,
)
from .kie_client import KieClient

__all__ = [
    "KieClient",
    "KieError",
    "KieAuthError",
    "KieTimeoutError",
    "KieTaskFailedError",
    "KieRateLimitError",
    "KieValidationError",
    "get_api_key",
    "get_base_url",
    "get_auth_token",
    "get_user_id",
    "is_proxied",
    "download_bytes",
    "download_to_output",
    "first_url",
    "all_urls",
    "image_url_to_tensor",
    "images_urls_to_tensor",
]
