"""Exceptions for the Kie.ai client."""


class KieError(Exception):
    """Base exception for all Kie.ai client errors."""

    def __init__(self, message: str, code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code

    def __str__(self) -> str:
        if self.code is not None:
            return f"[{self.code}] {self.message}"
        return self.message


class KieAuthError(KieError):
    """Raised when the API key is missing, invalid, or revoked."""


class KieTimeoutError(KieError):
    """Raised when a task takes too long to complete."""


class KieTaskFailedError(KieError):
    """Raised when Kie.ai reports the task as failed."""

    def __init__(self, message: str, task_id: str | None = None) -> None:
        super().__init__(message)
        self.task_id = task_id


class KieRateLimitError(KieError):
    """Raised when hitting Kie.ai rate limits (HTTP 429)."""


class KieValidationError(KieError):
    """Raised when input validation fails before sending to Kie."""
