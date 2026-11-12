import asyncio
import os
import re
import time
from typing import AsyncGenerator, Optional

import anthropic
import openai
import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from src.pricing import PRICING_TABLE, calculate_cost_usd
from src.schemas import ChunkTelemetry, StreamChunk

# Explicitly load .env from project root without relying on stack-frame inspection
load_dotenv(".env", override=True)


class GatewayError(Exception):
    """Base exception for LLM Gateway errors."""


class AuthenticationError(GatewayError):
    """Raised when provider API credentials are missing or invalid."""


class RateLimitExceededError(GatewayError):
    """Raised when provider rate limits are exceeded."""


class GatewayProviderError(GatewayError):
    """Raised on upstream provider connection or API failures."""


def _sanitize_error_message(message: str) -> str:
    """Strips OpenAI/Anthropic (sk-...) and Google (AIza... / AQ....) keys from strings."""
    sanitized = re.sub(r"sk-[a-zA-Z0-9_\-]+", "[REDACTED_API_KEY]", str(message))
    sanitized = re.sub(r"AIza[0-9A-Za-z\-_]{35}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"AQ\.[0-9A-Za-z\-_]+", "[REDACTED_API_KEY]", sanitized)
    return sanitized


class UnifiedLLMGateway:
    """Async multi-provider LLM gateway with persistent clients and real-time telemetry."""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ) -> None:
        self._openai_key = (
            openai_api_key if openai_api_key is not None else os.getenv("OPENAI_API_KEY", "")
        ).strip()
        self._anthropic_key = (
            anthropic_api_key if anthropic_api_key is not None else os.getenv("ANTHROPIC_API_KEY", "")
        ).strip()
        self._gemini_key = (
            gemini_api_key if gemini_api_key is not None else os.getenv("GEMINI_API_KEY", "")
        ).strip()

        # Prevent stale OS variables from hijacking google-genai auth
        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

        # Persistent async clients for connection pooling
        self._openai_client: Optional[openai.AsyncOpenAI] = (
            openai.AsyncOpenAI(api_key=self._openai_key) if self._openai_key else None
        )
        self._anthropic_client: Optional[anthropic.AsyncAnthropic] = (
            anthropic.AsyncAnthropic(api_key=self._anthropic_key) if self._anthropic_key else None
        )
        self._gemini_client: Optional[genai.Client] = (
            genai.Client(
                api_key=self._gemini_key,
                vertexai=False,
                http_options={"headers": {"x-goog-api-key": self._gemini_key}},
            )
            if self._gemini_key
            else None
        )

    def count_tokens_local(self, text: str, model: str = "gpt-4o") -> int:
        """Deterministic local token count using tiktoken with an offline fallback."""
        try:
            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            # Offline fallback when openaipublic.blob.core.windows.net is unreachable
            words = text.split()
            return max(1, len(words)) if words else 0

    def _validate_credentials(self, provider: str) -> None:
        if provider == "openai":
            if not self._openai_key or not self._openai_key.startswith("sk-") or self._openai_client is None:
                raise AuthenticationError(
                    "OpenAI API key is missing or malformed. Set a valid OPENAI_API_KEY starting with 'sk-'."
                )
        elif provider == "anthropic":
            if (
                not self._anthropic_key
                or not self._anthropic_key.startswith("sk-ant-")
                or self._anthropic_client is None
            ):
                raise AuthenticationError(
                    "Anthropic API key is missing or malformed. Set a valid ANTHROPIC_API_KEY starting with 'sk-ant-'."
                )
        elif provider == "gemini":
            if (
                not self._gemini_key
                or not self._gemini_key.startswith(("AIza", "AQ."))
                or self._gemini_client is None
            ):
                raise AuthenticationError(
                    "Gemini API key is missing or malformed. Set a valid GEMINI_API_KEY starting with 'AIza' or 'AQ.'."
                )

    def _build_final_chunk(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        start_time: float,
        first_token_time: Optional[float],
    ) -> StreamChunk:
        end_time = time.perf_counter()
        total_latency = max(end_time - start_time, 1e-6)
        ttft_ms = max(round(((first_token_time or end_time) - start_time) * 1000, 2), 0.01)
        tps = round(output_tokens / total_latency, 2)
        cost = calculate_cost_usd(model, input_tokens, output_tokens)

        return StreamChunk(
            delta_text="",
            provider=provider,  # type: ignore[arg-type]
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                ttft_ms=ttft_ms,
                total_latency_s=round(total_latency, 4),
                tokens_per_second=tps,
                total_cost_usd=cost,
            ),
        )

    async def stream(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
    ) -> AsyncGenerator[StreamChunk, None]:
        if model not in PRICING_TABLE:
            raise GatewayProviderError(f"Model '{model}' is not supported.")

        provider = PRICING_TABLE[model]["provider"]
        self._validate_credentials(provider)

        if provider == "openai":
            async for chunk in self._stream_openai(prompt, model, temperature):
                yield chunk
        elif provider == "anthropic":
            async for chunk in self._stream_anthropic(prompt, model, temperature):
                yield chunk
        elif provider == "gemini":
            async for chunk in self._stream_gemini(prompt, model, temperature):
                yield chunk

    async def _stream_openai(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        input_tokens = self.count_tokens_local(prompt, model)
        output_tokens = 0

        try:
            response_stream = await self._openai_client.chat.completions.create(  # type: ignore[union-attr]
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in response_stream:
                if not chunk.choices:
                    if chunk.usage:
                        input_tokens = chunk.usage.prompt_tokens
                        output_tokens = chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    yield StreamChunk(
                        delta_text=delta,
                        provider="openai",
                        model=model,
                        is_final=False,
                    )

        except openai.AuthenticationError as exc:
            raise AuthenticationError(
                f"OpenAI authentication failed: {_sanitize_error_message(str(exc))}"
            ) from None
        except openai.RateLimitError as exc:
            raise RateLimitExceededError(
                f"OpenAI rate limit exceeded: {_sanitize_error_message(str(exc))}"
            ) from None
        except openai.APIError as exc:
            raise GatewayProviderError(
                f"OpenAI upstream error: {_sanitize_error_message(str(exc))}"
            ) from None

        yield self._build_final_chunk(
            "openai", model, input_tokens, output_tokens, start_time, first_token_time
        )

    async def _stream_anthropic(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        input_tokens = self.count_tokens_local(prompt)
        output_tokens = 0

        try:
            async with self._anthropic_client.messages.stream(  # type: ignore[union-attr]
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        input_tokens = event.message.usage.input_tokens
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text") and event.delta.text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            yield StreamChunk(
                                delta_text=event.delta.text,
                                provider="anthropic",
                                model=model,
                                is_final=False,
                            )
                    elif event.type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            output_tokens = event.usage.output_tokens

        except anthropic.AuthenticationError as exc:
            raise AuthenticationError(
                f"Anthropic authentication failed: {_sanitize_error_message(str(exc))}"
            ) from None
        except anthropic.RateLimitError as exc:
            raise RateLimitExceededError(
                f"Anthropic rate limit exceeded: {_sanitize_error_message(str(exc))}"
            ) from None
        except anthropic.APIError as exc:
            raise GatewayProviderError(
                f"Anthropic upstream error: {_sanitize_error_message(str(exc))}"
            ) from None

        yield self._build_final_chunk(
            "anthropic", model, input_tokens, output_tokens, start_time, first_token_time
        )

    async def _stream_gemini(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        input_tokens = self.count_tokens_local(prompt)
        output_tokens = 0

        max_retries = 3
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response_stream = await self._gemini_client.aio.models.generate_content_stream(  # type: ignore[union-attr]
                    model=model,
                    contents=prompt,
                    config={"temperature": temperature},
                )

                async for chunk in response_stream:
                    text_delta = getattr(chunk, "text", None)
                    if text_delta:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        yield StreamChunk(
                            delta_text=text_delta,
                            provider="gemini",
                            model=model,
                            is_final=False,
                        )
                    usage = getattr(chunk, "usage_metadata", None)
                    if usage:
                        if getattr(usage, "prompt_token_count", None):
                            input_tokens = usage.prompt_token_count
                        if getattr(usage, "candidates_token_count", None):
                            output_tokens = usage.candidates_token_count

                break

            except genai_errors.APIError as exc:
                if exc.code in (401, 403):
                    raise AuthenticationError(
                        f"Gemini authentication failed: {_sanitize_error_message(str(exc))}"
                    ) from None
                if exc.code == 429:
                    raise RateLimitExceededError(
                        f"Gemini rate limit exceeded: {_sanitize_error_message(str(exc))}"
                    ) from None
                if exc.code == 503 and first_token_time is None and attempt < max_retries - 1:
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue
                raise GatewayProviderError(
                    f"Gemini upstream error: {_sanitize_error_message(str(exc))}"
                ) from None

        yield self._build_final_chunk(
            "gemini", model, input_tokens, output_tokens, start_time, first_token_time
        )

    async def close(self) -> None:
        if self._openai_client:
            await self._openai_client.close()
        if self._anthropic_client:
            await self._anthropic_client.close()