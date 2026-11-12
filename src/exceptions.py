"""src/exceptions.py - Typed domain exceptions and API credential redaction."""

import re
from typing import Any

_SECRET_PATTERN = re.compile(
    r"(?:sk-ant-[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+|AIzaSy[A-Za-z0-9_-]+|AQ\.[0-9A-Za-z\-_]+)"
)


def sanitize_message(message: Any) -> str:
    """Scrubs provider API keys (OpenAI, Anthropic, Gemini) from error strings."""
    return _SECRET_PATTERN.sub("[REDACTED_API_KEY]", str(message))


class GatewayError(Exception):
    """Base exception for all unified LLM gateway failures."""


class AuthenticationError(GatewayError):
    """Raised when an API key is missing, malformed, or rejected by the provider."""


class RateLimitExceededError(GatewayError):
    """Raised when an upstream provider enforces HTTP 429 rate or quota limits."""


class GatewayProviderError(GatewayError):
    """Raised on upstream network failures, model deprecations, or unexpected API errors."""
