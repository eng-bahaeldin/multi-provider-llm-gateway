"""src/gateway.py - Unified multi-provider asynchronous streaming gateway."""

import asyncio
from functools import lru_cache
import os
import random
import re
import sys
import time
from typing import AsyncGenerator, Optional

import anthropic
import httpx
import openai
import tiktoken
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors

from src.exceptions import (
    AuthenticationError,
    GatewayError,
    GatewayProviderError,
    RateLimitExceededError,
    sanitize_message,
)
from src.pricing import PRICING_TABLE_PER_1M, calculate_cost
from src.schemas import ChunkTelemetry, Provider, StreamChunk

__all__ = [
    "UnifiedLLMGateway",
    "GatewayError",
    "AuthenticationError",
    "RateLimitExceededError",
    "GatewayProviderError",
    "sanitize_message",
]

load_dotenv(".env", override=True)

if sys.platform == "win32":
    try:
        if isinstance(asyncio.get_event_loop_policy(), asyncio.WindowsProactorEventLoopPolicy):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

_POOL_LIMITS = httpx.Limits(
    max_keepalive_connections=10,
    max_connections=20,
    keepalive_expiry=30.0,
)


@lru_cache(maxsize=4)
def _get_cached_encoding(model: str) -> tiktoken.Encoding:
    """Caches tiktoken BPE encodings in memory to eliminate per-request initialization overhead."""
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _sanitize_error_message(message: object) -> str:
    """Strips OpenAI/Anthropic (sk-...) and Google (AIza... / AQ....) keys from strings."""
    sanitized = sanitize_message(message)
    sanitized = re.sub(r"AQ\.[0-9A-Za-z\-_]+", "[REDACTED_API_KEY]", sanitized)
    return sanitized


class UnifiedLLMGateway:
    """Async multi-provider LLM gateway with connection pooling and real-time telemetry."""

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

        os.environ.pop("GOOGLE_API_KEY", None)
        os.environ.pop("GOOGLE_GENAI_USE_VERTEXAI", None)

    @property
    def _openai_client(self) -> Optional[openai.AsyncOpenAI]:
        if "_openai_client" not in self.__dict__ and self._openai_key:
            self.__dict__["_openai_client"] = openai.AsyncOpenAI(
                api_key=self._openai_key,
                http_client=httpx.AsyncClient(limits=_POOL_LIMITS, timeout=60.0),
            )
        return self.__dict__.get("_openai_client")

    @property
    def _anthropic_client(self) -> Optional[anthropic.AsyncAnthropic]:
        if "_anthropic_client" not in self.__dict__ and self._anthropic_key:
            self.__dict__["_anthropic_client"] = anthropic.AsyncAnthropic(
                api_key=self._anthropic_key,
            )
        return self.__dict__.get("_anthropic_client")

    @property
    def _gemini_client(self) -> Optional[genai.Client]:
        if "_gemini_client" not in self.__dict__ and self._gemini_key:
            self.__dict__["_gemini_client"] = genai.Client(
                api_key=self._gemini_key,
                vertexai=False,
                http_options={"headers": {"x-goog-api-key": self._gemini_key}},
            )
        return self.__dict__.get("_gemini_client")

    def count_tokens_local(self, text: str, model: str = "gpt-4o") -> int:
        """Deterministic local token count using cached tiktoken with an offline fallback."""
        try:
            encoding = _get_cached_encoding(model)
            return len(encoding.encode(text))
        except Exception:
            words = text.split()
            return max(1, len(words)) if words else 0

    def estimate_prompt_tokens(self, prompt: str) -> int:
        """Alias for deterministic local prompt token estimation."""
        return self.count_tokens_local(prompt)

    def _validate_credentials(self, provider: Provider) -> None:
        if provider == Provider.OPENAI and "_openai_client" not in self.__dict__:
            if not self._openai_key or not self._openai_key.startswith("sk-"):
                raise AuthenticationError(
                    "OpenAI API key is missing or malformed. Set a valid OPENAI_API_KEY starting with 'sk-'."
                )
        elif provider == Provider.ANTHROPIC and "_anthropic_client" not in self.__dict__:
            if not self._anthropic_key or not self._anthropic_key.startswith("sk-ant-"):
                raise AuthenticationError(
                    "Anthropic API key is missing or malformed. Set a valid ANTHROPIC_API_KEY starting with 'sk-ant-'."
                )
        elif provider == Provider.GEMINI and "_gemini_client" not in self.__dict__:
            if not self._gemini_key or not self._gemini_key.startswith(("AIza", "AQ.")):
                raise AuthenticationError(
                    "Gemini API key is missing or malformed. Set a valid GEMINI_API_KEY starting with 'AIza' or 'AQ.'."
                )

    def _resolve_provider(self, model: str) -> Provider:
        model_lower = model.lower()
        if "gemini" in model_lower:
            return Provider.GEMINI
        if "claude" in model_lower:
            return Provider.ANTHROPIC
        if model in PRICING_TABLE_PER_1M or model_lower.startswith(("gpt-", "o1", "o3")):
            return Provider.OPENAI
        raise GatewayProviderError(f"Model '{model}' is not supported.")

    def _build_final_chunk(
        self,
        provider: Provider,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        start_time: float,
        first_token_time: Optional[float],
    ) -> StreamChunk:
        end_time = time.perf_counter()
        total_latency = max(end_time - start_time, 1e-6)
        ttft_ms = max(round(((first_token_time or end_time) - start_time) * 1000, 2), 0.01)
        tps = round(completion_tokens / total_latency, 2)
        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        return StreamChunk(
            delta_text="",
            provider=provider,
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                time_to_first_token_ms=ttft_ms,
                total_latency_seconds=max(round(total_latency, 4), 0.0001),
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
        provider = self._resolve_provider(model)
        self._validate_credentials(provider)

        if provider == Provider.OPENAI:
            async for chunk in self._stream_openai(prompt, model, temperature):
                yield chunk
        elif provider == Provider.ANTHROPIC:
            async for chunk in self._stream_anthropic(prompt, model, temperature):
                yield chunk
        elif provider == Provider.GEMINI:
            async for chunk in self._stream_gemini(prompt, model, temperature):
                yield chunk

    async def _stream_openai(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._openai_client
        if client is None:
            raise AuthenticationError("OPENAI_API_KEY is not configured.")

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prompt_tokens = self.count_tokens_local(prompt, model)
        completion_tokens = 0
        response_stream = None

        try:
            response_stream = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in response_stream:
                if not chunk.choices:
                    if chunk.usage:
                        prompt_tokens = chunk.usage.prompt_tokens
                        completion_tokens = chunk.usage.completion_tokens
                    continue

                delta = chunk.choices[0].delta.content
                if delta:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    if not chunk.usage:
                        completion_tokens += 1
                    yield StreamChunk(
                        delta_text=delta,
                        provider=Provider.OPENAI,
                        model=model,
                        is_final=False,
                    )

        except openai.AuthenticationError as exc:
            raise AuthenticationError(
                f"OpenAI authentication failed: {_sanitize_error_message(exc)}"
            ) from None
        except openai.RateLimitError as exc:
            raise RateLimitExceededError(
                f"OpenAI rate limit exceeded: {_sanitize_error_message(exc)}"
            ) from None
        except openai.APIError as exc:
            raise GatewayProviderError(
                f"OpenAI upstream error: {_sanitize_error_message(exc)}"
            ) from None
        finally:
            if response_stream is not None and hasattr(response_stream, "aclose"):
                try:
                    await response_stream.aclose()
                except Exception:
                    pass

        yield self._build_final_chunk(
            Provider.OPENAI, model, prompt_tokens, completion_tokens, start_time, first_token_time
        )

    async def _stream_anthropic(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._anthropic_client
        if client is None:
            raise AuthenticationError("ANTHROPIC_API_KEY is not configured.")

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prompt_tokens = self.count_tokens_local(prompt)
        completion_tokens = 0

        try:
            async with client.messages.stream(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                async for event in stream:
                    if event.type == "message_start":
                        prompt_tokens = event.message.usage.input_tokens
                    elif event.type == "content_block_delta":
                        if hasattr(event.delta, "text") and event.delta.text:
                            if first_token_time is None:
                                first_token_time = time.perf_counter()
                            yield StreamChunk(
                                delta_text=event.delta.text,
                                provider=Provider.ANTHROPIC,
                                model=model,
                                is_final=False,
                            )
                    elif event.type == "message_delta":
                        if hasattr(event, "usage") and event.usage:
                            completion_tokens = event.usage.output_tokens

        except anthropic.AuthenticationError as exc:
            raise AuthenticationError(
                f"Anthropic authentication failed: {_sanitize_error_message(exc)}"
            ) from None
        except anthropic.RateLimitError as exc:
            raise RateLimitExceededError(
                f"Anthropic rate limit exceeded: {_sanitize_error_message(exc)}"
            ) from None
        except anthropic.APIError as exc:
            raise GatewayProviderError(
                f"Anthropic upstream error: {_sanitize_error_message(exc)}"
            ) from None

        yield self._build_final_chunk(
            Provider.ANTHROPIC, model, prompt_tokens, completion_tokens, start_time, first_token_time
        )

    async def _stream_gemini(
        self, prompt: str, model: str, temperature: float
    ) -> AsyncGenerator[StreamChunk, None]:
        client = self._gemini_client
        if client is None:
            raise AuthenticationError("GEMINI_API_KEY is not configured.")

        start_time = time.perf_counter()
        first_token_time: Optional[float] = None
        prompt_tokens = self.count_tokens_local(prompt)
        completion_tokens = 0
        streamed_chunks_count = 0
        response_stream = None
        active_model = model

        candidate_models = [model]
        for fallback in ("gemini-3.8-flash", "gemini-3.5-flash", "gemini-3.8-pro", "gemini-3.5-pro"):
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        try:
            stream_completed = False
            last_429_error: Optional[genai_errors.APIError] = None
            last_upstream_error: Optional[genai_errors.APIError] = None

            for candidate in candidate_models:
                for attempt in range(2):
                    try:
                        response_stream = await client.aio.models.generate_content_stream(
                            model=candidate,
                            contents=prompt,
                            config={"temperature": temperature},
                        )

                        async for chunk in response_stream:
                            text_delta = ""
                            try:
                                text_delta = getattr(chunk, "text", None) or ""
                            except Exception:
                                text_delta = ""

                            if text_delta:
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                streamed_chunks_count += 1
                                active_model = candidate
                                yield StreamChunk(
                                    delta_text=text_delta,
                                    provider=Provider.GEMINI,
                                    model=candidate,
                                    is_final=False,
                                )

                            usage = getattr(chunk, "usage_metadata", None)
                            if usage:
                                if getattr(usage, "prompt_token_count", None):
                                    prompt_tokens = usage.prompt_token_count
                                if getattr(usage, "candidates_token_count", None):
                                    completion_tokens = usage.candidates_token_count

                        stream_completed = True
                        break

                    except genai_errors.APIError as exc:
                        if exc.code in (401, 403):
                            raise AuthenticationError(
                                f"Gemini authentication failed: {_sanitize_error_message(exc)}"
                            ) from None
                        if exc.code == 429 and first_token_time is None:
                            last_429_error = exc
                            break  # Immediately fail over to the next Gemini model's quota pool
                        if exc.code == 503 and first_token_time is None:
                            last_upstream_error = exc
                            await asyncio.sleep(0.3 * (attempt + 1) + random.uniform(0.05, 0.2))
                            continue
                        if exc.code == 404 and candidate != model and first_token_time is None:
                            break  # Skip unprovisioned fallback model
                        raise GatewayProviderError(
                            f"Gemini upstream error: {_sanitize_error_message(exc)}"
                        ) from None

                if stream_completed:
                    break

            if not stream_completed:
                if last_429_error is not None:
                    raise RateLimitExceededError(
                        f"Gemini rate limit exceeded: {_sanitize_error_message(last_429_error)}"
                    ) from None
                if last_upstream_error is not None:
                    raise GatewayProviderError(
                        f"Gemini upstream error: {_sanitize_error_message(last_upstream_error)}"
                    ) from None

        finally:
            if response_stream is not None and hasattr(response_stream, "aclose"):
                try:
                    await response_stream.aclose()
                except Exception:
                    pass

        if completion_tokens == 0 and streamed_chunks_count > 0:
            completion_tokens = streamed_chunks_count

        yield self._build_final_chunk(
            Provider.GEMINI, active_model, prompt_tokens, completion_tokens, start_time, first_token_time
        )

    async def close(self) -> None:
        if "_openai_client" in self.__dict__ and self.__dict__["_openai_client"]:
            await self.__dict__["_openai_client"].close()
        if "_anthropic_client" in self.__dict__ and self.__dict__["_anthropic_client"]:
            await self.__dict__["_anthropic_client"].close()
