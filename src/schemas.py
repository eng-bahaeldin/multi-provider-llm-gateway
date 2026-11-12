from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChunkTelemetry(BaseModel):
    """Execution telemetry emitted for each final response chunk."""

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    ttft_ms: float = Field(default=0.0, ge=0.0)
    total_time_sec: float = Field(default=0.0, ge=0.0)
    total_latency_s: float = Field(default=0.0, ge=0.0)
    tokens_per_second: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)


class StreamChunk(BaseModel):
    """Normalized stream chunk produced by the gateway."""

    delta_text: str = Field(default="")
    provider: Literal["openai", "anthropic", "gemini"]
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None


class GatewayError(Exception):
    """Base exception for all gateway failures."""


class AuthenticationError(GatewayError):
    """Raised when provider credentials are missing or rejected."""


class RateLimitError(GatewayError):
    """Raised when the upstream provider reports rate limiting or temporary capacity issues."""


RateLimitExceededError = RateLimitError
