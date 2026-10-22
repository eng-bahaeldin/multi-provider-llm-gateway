from typing import Optional
from pydantic import BaseModel, Field


class ChunkTelemetry(BaseModel):
    """Execution telemetry emitted exclusively on the final chunk."""
    prompt_tokens: int = Field(default=0, description="Input tokens counted")
    completion_tokens: int = Field(default=0, description="Generated output tokens")
    total_tokens: int = Field(default=0, description="Sum of prompt and completion tokens")
    ttft_ms: float = Field(default=0.0, description="Time to first token in milliseconds")
    total_time_sec: float = Field(default=0.0, description="Total wall-clock duration in seconds")
    tokens_per_second: float = Field(default=0.0, description="Throughput generation speed")
    total_cost_usd: float = Field(default=0.0, description="Micro-cent cost accounting in USD")


class StreamChunk(BaseModel):
    """Normalized delta chunk emitted to the consumer on every iteration."""
    delta_text: str = Field(default="", description="Incremental text fragment")
    provider: str = Field(..., description="Provider identifier ('gemini', 'openai', or 'anthropic')")
    model: str = Field(..., description="Active model identifier")
    is_final: bool = Field(default=False, description="True only on terminal chunk")
    telemetry: Optional[ChunkTelemetry] = Field(default=None, description="Populated only when is_final=True")


# ==============================================================================
# Domain-Specific Exception Hierarchy
# ==============================================================================

class GatewayError(Exception):
    """Base exception for all gateway operational failures."""
    pass


class AuthenticationError(GatewayError):
    """Raised when an API key is missing, invalid, or rejected."""
    pass


class RateLimitError(GatewayError):
    """Raised when upstream API rate limits are reached."""
    pass