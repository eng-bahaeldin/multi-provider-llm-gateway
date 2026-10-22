import asyncio
import os
import random
import re
import time
from typing import AsyncGenerator
import tiktoken
from dotenv import load_dotenv

# Provider SDKs & Errors
from google import genai
from google.genai import types
from google.genai.errors import APIError as GoogleAPIError
from openai import (
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimit,
    APIStatusError as OpenAIStatusError,
)
from anthropic import (
    AsyncAnthropic,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimit,
    APIStatusError as AnthropicStatusError,
)

from src.schemas import StreamChunk, ChunkTelemetry, GatewayError, AuthenticationError, RateLimitError
from src.pricing import calculate_cost

load_dotenv(override=False)

# ------------------------------------------------------------------------------
# Model Fallback Directory
# Automatically reroutes traffic when preview endpoints experience demand spikes
# ------------------------------------------------------------------------------
FALLBACK_MODELS: dict[str, str] = {
    "gemini-3.8-flash": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-1.5-flash",
}


def sanitize_message(msg: str) -> str:
    """Masks secret API keys to prevent credential leakage in logs or exceptions."""
    sanitized = re.sub(r'(sk-[a-zA-Z0-9_\-]{20,})', '[REDACTED_KEY]', str(msg))
    sanitized = re.sub(r'(AIzaSy[a-zA-Z0-9_\-]{33})', '[REDACTED_GEMINI_KEY]', sanitized)
    sanitized = re.sub(r'x-api-key:[a-zA-Z0-9_\-]+', 'x-api-key:[REDACTED_KEY]', sanitized)
    return sanitized


def is_transient_error(err: Exception) -> bool:
    """Detects HTTP 429 (rate limits) and HTTP 503 (server demand spikes)."""
    if isinstance(err, GoogleAPIError):
        code = getattr(err, "code", None)
        if code in (429, 503):
            return True

    if isinstance(err, (OpenAIRateLimit, AnthropicRateLimit)):
        return True

    if isinstance(err, (OpenAIStatusError, AnthropicStatusError)):
        if err.status_code in (429, 500, 502, 503):
            return True

    err_str = str(err).lower()
    return any(marker in err_str for marker in ("503", "unavailable", "high demand", "429", "rate limit"))


class UnifiedLLMGateway:
    """Multi-provider streaming gateway with automatic retry and model fallback."""

    def __init__(
        self,
        gemini_api_key: str | None = None,
        openai_api_key: str | None = None,
        anthropic_api_key: str | None = None,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ):
        self.gemini_key = gemini_api_key or os.getenv("GEMINI_API_KEY")
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.anthropic_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        # Persistent connection-pooled client singletons
        self._gemini_client: genai.Client | None = None
        self._openai_client: AsyncOpenAI | None = None
        self._anthropic_client: AsyncAnthropic | None = None

    @property
    def gemini_client(self) -> genai.Client:
        if not self.gemini_key:
            raise AuthenticationError("GEMINI_API_KEY is not configured or missing.")
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=self.gemini_key)
        return self._gemini_client

    @property
    def openai_client(self) -> AsyncOpenAI:
        if not self.openai_key:
            raise AuthenticationError("OPENAI_API_KEY is not configured or missing.")
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(api_key=self.openai_key)
        return self._openai_client

    @property
    def anthropic_client(self) -> AsyncAnthropic:
        if not self.anthropic_key:
            raise AuthenticationError("ANTHROPIC_API_KEY is not configured or missing.")
        if self._anthropic_client is None:
            self._anthropic_client = AsyncAnthropic(api_key=self.anthropic_key)
        return self._anthropic_client

    def count_prompt_tokens(self, text: str, model: str) -> int:
        """Deterministic prompt token counting via tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    async def stream(
        self, prompt: str, model: str, temperature: float = 0.7
    ) -> AsyncGenerator[StreamChunk, None]:
        """Streams normalized chunks, handling retries and fallback before first yield."""
        provider = "gemini" if "gemini" in model else ("openai" if ("gpt" in model or "o1" in model) else "anthropic")

        for attempt in range(self.max_retries + 1):
            start_time = time.perf_counter()
            first_token_time: float | None = None
            accumulated_text = ""
            prompt_tokens = 0
            completion_tokens = 0
            yielded_any_chunk = False

            try:
                # -------------------------------------------------------------
                # 1. Google Gemini Branch
                # -------------------------------------------------------------
                if provider == "gemini":
                    client = self.gemini_client
                    prompt_tokens = self.count_prompt_tokens(prompt, model)
                    config = types.GenerateContentConfig(temperature=temperature)

                    response = await client.aio.models.generate_content_stream(
                        model=model,
                        contents=prompt,
                        config=config,
                    )

                    async for chunk in response:
                        if chunk.usage_metadata:
                            prompt_tokens = chunk.usage_metadata.prompt_token_count or prompt_tokens
                            completion_tokens = chunk.usage_metadata.candidates_token_count or completion_tokens

                        delta = chunk.text or ""
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            accumulated_text += delta
                            yielded_any_chunk = True
                            yield StreamChunk(
                                delta_text=delta,
                                provider="gemini",
                                model=model,
                                is_final=False,
                            )

                # -------------------------------------------------------------
                # 2. OpenAI Branch
                # -------------------------------------------------------------
                elif provider == "openai":
                    client = self.openai_client
                    prompt_tokens = self.count_prompt_tokens(prompt, model)

                    response = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        stream=True,
                        stream_options={"include_usage": True},
                    )

                    async for chunk in response:
                        if chunk.usage is not None:
                            prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                            completion_tokens = chunk.usage.completion_tokens or completion_tokens

                        if chunk.choices and len(chunk.choices) > 0:
                            delta = chunk.choices[0].delta.content or ""
                            if delta:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                accumulated_text += delta
                                yielded_any_chunk = True
                                yield StreamChunk(
                                    delta_text=delta,
                                    provider="openai",
                                    model=model,
                                    is_final=False,
                                )

                # -------------------------------------------------------------
                # 3. Anthropic Branch
                # -------------------------------------------------------------
                elif provider == "anthropic":
                    client = self.anthropic_client
                    async with client.messages.stream(
                        model=model,
                        max_tokens=2048,
                        temperature=temperature,
                        messages=[{"role": "user", "content": prompt}],
                    ) as stream_ctx:
                        async for event in stream_ctx:
                            if event.type == "message_start":
                                prompt_tokens = event.message.usage.input_tokens
                            elif event.type == "content_block_delta":
                                delta = event.delta.text
                                if delta:
                                    if first_token_time is None:
                                        first_token_time = time.perf_counter()
                                    accumulated_text += delta
                                    yielded_any_chunk = True
                                    yield StreamChunk(
                                        delta_text=delta,
                                        provider="anthropic",
                                        model=model,
                                        is_final=False,
                                    )
                            elif event.type == "message_delta":
                                if hasattr(event.usage, "output_tokens") and event.usage.output_tokens:
                                    completion_tokens = event.usage.output_tokens

                # -------------------------------------------------------------
                # Terminal Telemetry Receipt
                # -------------------------------------------------------------
                end_time = time.perf_counter()
                total_duration = end_time - start_time
                ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0

                if completion_tokens == 0:
                    completion_tokens = self.count_prompt_tokens(accumulated_text, model)

                gen_window = (end_time - first_token_time) if first_token_time else total_duration
                tps = (completion_tokens / gen_window) if gen_window > 0 else 0.0
                cost = calculate_cost(model, prompt_tokens, completion_tokens)

                final_telemetry = ChunkTelemetry(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                    ttft_ms=round(ttft_ms, 2),
                    total_time_sec=round(total_duration, 3),
                    tokens_per_second=round(tps, 2),
                    total_cost_usd=cost,
                )

                yield StreamChunk(
                    delta_text="",
                    provider=provider,
                    model=model,
                    is_final=True,
                    telemetry=final_telemetry,
                )
                return  # Stream completed cleanly

            except Exception as err:
                # Mid-stream error: cannot retry or fallback without duplicating output
                if yielded_any_chunk:
                    raise GatewayError(sanitize_message(f"Stream interrupted mid-generation: {err}")) from err

                # Case A: Retry with exponential backoff on current model
                if is_transient_error(err) and attempt < self.max_retries:
                    delay = min(self.base_delay * (2 ** attempt) + random.uniform(0.1, 0.4), self.max_delay)
                    await asyncio.sleep(delay)
                    continue

                # Case B: Retries exhausted -> Cascade to fallback model
                if is_transient_error(err):
                    fallback_model = FALLBACK_MODELS.get(model)
                    if fallback_model and fallback_model != model:
                        async for fallback_chunk in self.stream(
                            prompt=prompt,
                            model=fallback_model,
                            temperature=temperature,
                        ):
                            yield fallback_chunk
                        return

                # Case C: Permanent errors or exhausted fallbacks
                if isinstance(err, (OpenAIAuthError, AnthropicAuthError)):
                    raise AuthenticationError(sanitize_message(f"Authentication failed: {err}")) from err
                elif isinstance(err, GoogleAPIError) and getattr(err, "code", None) in (401, 403):
                    raise AuthenticationError(sanitize_message(f"Gemini authentication failed: {err}")) from err
                elif is_transient_error(err):
                    raise RateLimitError(sanitize_message(f"Capacity exhausted after retries and fallback: {err}")) from err
                else:
                    raise GatewayError(sanitize_message(f"Gateway execution error: {err}")) from err