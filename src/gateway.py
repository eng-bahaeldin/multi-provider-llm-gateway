import asyncio
import os
import random
import re
import time
from typing import AsyncGenerator, Optional

import tiktoken
from anthropic import (
    APIStatusError as AnthropicStatusError,
    AsyncAnthropic,
    AuthenticationError as AnthropicAuthError,
    RateLimitError as AnthropicRateLimit,
)
from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError as GoogleAPIError
from openai import (
    APIStatusError as OpenAIStatusError,
    AsyncOpenAI,
    AuthenticationError as OpenAIAuthError,
    RateLimitError as OpenAIRateLimit,
)

from src.pricing import PRICING_TABLE, calculate_cost, calculate_cost_usd
from src.schemas import (
    AuthenticationError,
    ChunkTelemetry,
    GatewayError,
    RateLimitError,
    StreamChunk,
)

load_dotenv(override=False)

FALLBACK_MODELS: dict[str, str] = {
    "gemini-3.8-flash": "gemini-2.0-flash",
    "gemini-2.0-flash": "gemini-1.5-flash",
}


def sanitize_message(msg: str) -> str:
    """Mask secret API keys before raising or logging gateway errors."""
    sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]{20,}", "[REDACTED_API_KEY]", str(msg))
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"AQ\.[0-9A-Za-z\-_]+", "[REDACTED_API_KEY]", sanitized)
    return sanitized


def is_transient_error(err: Exception) -> bool:
    """Detect transient upstream errors such as rate limits or service outages."""
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
    """Async multi-provider LLM gateway with provider-agnostic streaming."""

    def __init__(
        self,
        gemini_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        max_retries: int = 2,
        base_delay: float = 1.0,
        max_delay: float = 8.0,
    ) -> None:
        self.gemini_key = (gemini_api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.openai_key = (openai_api_key or os.getenv("OPENAI_API_KEY") or "").strip()
        self.anthropic_key = (anthropic_api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()

        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

        self._openai_client: Optional[AsyncOpenAI] = AsyncOpenAI(api_key=self.openai_key) if self.openai_key else None
        self._anthropic_client: Optional[AsyncAnthropic] = (
            AsyncAnthropic(api_key=self.anthropic_key) if self.anthropic_key else None
        )
        self._gemini_client: Optional[genai.Client] = (
            genai.Client(api_key=self.gemini_key) if self.gemini_key else None
        )

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
        """Deterministic prompt-token count using tiktoken when available."""
        try:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            return max(1, len(text.split())) if text else 0

    def count_tokens_local(self, text: str, model: str = "gpt-4o") -> int:
        return self.count_prompt_tokens(text, model)

    def _resolve_provider(self, model: str) -> str:
        model_name = model.lower()
        if "gemini" in model_name:
            return "gemini"
        if "gpt" in model_name or "o1" in model_name or "o3" in model_name:
            return "openai"
        if "claude" in model_name:
            return "anthropic"
        raise GatewayError(f"Unsupported model '{model}'.")

    def _make_final_chunk(
        self,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        start_time: float,
        first_token_time: Optional[float],
    ) -> StreamChunk:
        total_duration = time.perf_counter() - start_time
        ttft_ms = ((first_token_time - start_time) * 1000.0) if first_token_time else 0.0
        throughput = (completion_tokens / max(total_duration, 1e-6)) if completion_tokens else 0.0
        cost = calculate_cost(model, prompt_tokens, completion_tokens)
        telemetry = ChunkTelemetry(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            ttft_ms=round(ttft_ms, 2),
            total_time_sec=round(total_duration, 3),
            total_latency_s=round(total_duration, 3),
            tokens_per_second=round(throughput, 2),
            total_cost_usd=cost,
        )
        return StreamChunk(delta_text="", provider=provider, model=model, is_final=True, telemetry=telemetry)

    async def _stream_openai(self, prompt: str, model: str, temperature: float) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prompt_tokens = self.count_prompt_tokens(prompt, model)
        completion_tokens = 0

        try:
            response = await self.openai_client.chat.completions.create(
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
                    delta = getattr(chunk.choices[0].delta, "content", None) or ""
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        yield StreamChunk(delta_text=delta, provider="openai", model=model, is_final=False)

        except OpenAIAuthError as exc:
            raise AuthenticationError(sanitize_message(f"OpenAI authentication failed: {exc}")) from None
        except OpenAIRateLimit as exc:
            raise RateLimitError(sanitize_message(f"OpenAI rate limit exceeded: {exc}")) from None
        except OpenAIStatusError as exc:
            if exc.status_code in (401, 403):
                raise AuthenticationError(sanitize_message(f"OpenAI authentication failed: {exc}")) from None
            if exc.status_code in (429, 500, 502, 503):
                raise RateLimitError(sanitize_message(f"OpenAI service rate limit or outage: {exc}")) from None
            raise GatewayError(sanitize_message(f"OpenAI upstream error: {exc}")) from None
        except Exception as exc:
            raise GatewayError(sanitize_message(f"OpenAI stream failed: {exc}")) from None

        yield self._make_final_chunk("openai", model, prompt_tokens, completion_tokens, start_time, first_token_time)

    async def _stream_anthropic(self, prompt: str, model: str, temperature: float) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prompt_tokens = self.count_prompt_tokens(prompt, model)
        completion_tokens = 0

        try:
            async with self.anthropic_client.messages.stream(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream_ctx:
                async for event in stream_ctx:
                    if event.type == "message_start":
                        prompt_tokens = getattr(event.message.usage, "input_tokens", prompt_tokens)
                    elif event.type == "content_block_delta":
                        delta = getattr(event.delta, "text", "") or ""
                        if delta:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            yield StreamChunk(delta_text=delta, provider="anthropic", model=model, is_final=False)
                    elif event.type == "message_delta":
                        completion_tokens = getattr(getattr(event, "usage", None), "output_tokens", completion_tokens)

        except AnthropicAuthError as exc:
            raise AuthenticationError(sanitize_message(f"Anthropic authentication failed: {exc}")) from None
        except AnthropicRateLimit as exc:
            raise RateLimitError(sanitize_message(f"Anthropic rate limit exceeded: {exc}")) from None
        except AnthropicStatusError as exc:
            if exc.status_code in (401, 403):
                raise AuthenticationError(sanitize_message(f"Anthropic authentication failed: {exc}")) from None
            if exc.status_code in (429, 500, 502, 503):
                raise RateLimitError(sanitize_message(f"Anthropic service rate limit or outage: {exc}")) from None
            raise GatewayError(sanitize_message(f"Anthropic upstream error: {exc}")) from None
        except Exception as exc:
            raise GatewayError(sanitize_message(f"Anthropic stream failed: {exc}")) from None

        yield self._make_final_chunk("anthropic", model, prompt_tokens, completion_tokens, start_time, first_token_time)

    async def _stream_gemini(self, prompt: str, model: str, temperature: float) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prompt_tokens = self.count_prompt_tokens(prompt, model)
        completion_tokens = 0

        for attempt in range(self.max_retries + 1):
            try:
                response = await self.gemini_client.aio.models.generate_content_stream(
                    model=model,
                    contents=prompt,
                    config={"temperature": temperature},
                )

                async for chunk in response:
                    delta = getattr(chunk, "text", "") or ""
                    if delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        yield StreamChunk(delta_text=delta, provider="gemini", model=model, is_final=False)

                    usage = getattr(chunk, "usage_metadata", None)
                    if usage is not None:
                        prompt_tokens = getattr(usage, "prompt_token_count", prompt_tokens) or prompt_tokens
                        completion_tokens = getattr(usage, "candidates_token_count", completion_tokens) or completion_tokens

                yield self._make_final_chunk("gemini", model, prompt_tokens, completion_tokens, start_time, first_token_time)
                return
            except GoogleAPIError as exc:
                code = getattr(exc, "code", None)
                if code in (401, 403):
                    raise AuthenticationError(sanitize_message(f"Gemini authentication failed: {exc}")) from None
                if code == 429:
                    raise RateLimitError(sanitize_message(f"Gemini rate limit exceeded: {exc}")) from None
                if code == 503 and attempt < self.max_retries:
                    await asyncio.sleep(min(self.base_delay * (2**attempt) + random.uniform(0.1, 0.4), self.max_delay))
                    continue
                raise GatewayError(sanitize_message(f"Gemini upstream error: {exc}")) from None
            except Exception as exc:
                if is_transient_error(exc) and attempt < self.max_retries:
                    await asyncio.sleep(min(self.base_delay * (2**attempt) + random.uniform(0.1, 0.4), self.max_delay))
                    continue
                raise GatewayError(sanitize_message(f"Gemini stream failed: {exc}")) from None

    async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:
        """Stream a prompt through the selected provider and yield normalized chunks."""
        provider = self._resolve_provider(model)

        for attempt in range(self.max_retries + 1):
            try:
                if provider == "openai":
                    async for chunk in self._stream_openai(prompt, model, temperature):
                        yield chunk
                    return
                if provider == "anthropic":
                    async for chunk in self._stream_anthropic(prompt, model, temperature):
                        yield chunk
                    return
                async for chunk in self._stream_gemini(prompt, model, temperature):
                    yield chunk
                return
            except (AuthenticationError, RateLimitError, GatewayError):
                raise
            except Exception as exc:
                if is_transient_error(exc) and attempt < self.max_retries:
                    await asyncio.sleep(min(self.base_delay * (2**attempt) + random.uniform(0.1, 0.4), self.max_delay))
                    continue
                raise GatewayError(sanitize_message(f"Gateway execution failed: {exc}")) from exc

        raise GatewayError(f"Gateway exhausted retries for model '{model}'.")

    async def close(self) -> None:
        if self._openai_client is not None:
            await self._openai_client.close()
        if self._anthropic_client is not None:
            await self._anthropic_client.close()


__all__ = ["UnifiedLLMGateway", "GatewayError", "AuthenticationError", "RateLimitError"]
