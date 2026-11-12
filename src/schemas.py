"""src/schemas.py - Standardized Pydantic v2 data models for multi-provider streaming."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field


class Provider(str, Enum):
    """Supported upstream LLM providers."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"


class ChunkTelemetry(BaseModel):
    """Immutable telemetry and micro-cent cost metrics emitted at stream completion."""

    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = Field(..., ge=0, description="Number of input prompt tokens")
    completion_tokens: int = Field(..., ge=0, description="Number of generated output tokens")
    total_tokens: int = Field(..., ge=0, description="Sum of prompt and completion tokens")
    time_to_first_token_ms: float = Field(..., ge=0.0, description="Latency to first non-empty token in ms")
    total_latency_seconds: float = Field(..., ge=0.0, description="Total end-to-end request duration in seconds")
    tokens_per_second: float = Field(..., ge=0.0, description="Generation throughput in tokens/sec")
    total_cost_usd: float = Field(..., ge=0.0, description="Total execution cost in USD (6 decimal places)")


class StreamChunk(BaseModel):
    """Normalized streaming packet yielded across all LLM providers."""

    model_config = ConfigDict(frozen=True)

    delta_text: str = Field(default="", description="Incremental text content delta")
    provider: Provider = Field(..., description="Normalized provider identifier")
    model: str = Field(..., min_length=1, description="Target model identifier")
    is_final: bool = Field(default=False, description="True only on the terminal telemetry packet")
    telemetry: Optional[ChunkTelemetry] = Field(
        default=None,
        description="Populated only when is_final is True",
    )
