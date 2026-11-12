from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChunkTelemetry(BaseModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    ttft_ms: float = Field(default=0.0, ge=0.0, description="Time to first non-empty token in ms")
    total_latency_s: float = Field(default=0.0, ge=0.0)
    tokens_per_second: float = Field(default=0.0, ge=0.0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)


class StreamChunk(BaseModel):
    delta_text: str = Field(default="")
    provider: Literal["openai", "anthropic", "gemini"]
    model: str
    is_final: bool = False
    telemetry: Optional[ChunkTelemetry] = None