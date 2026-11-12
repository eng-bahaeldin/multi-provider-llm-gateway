from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.gateway import UnifiedLLMGateway
from src.schemas import StreamChunk


class AsyncIteratorMock:
    def __init__(self, items):
        self._items = iter(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._items)
        except StopIteration:
            raise StopAsyncIteration


class AsyncContextManagerMock:
    def __init__(self, items):
        self._items = items

    async def __aenter__(self):
        return AsyncIteratorMock(self._items)

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_unified_stream_protocol():
    """Test 1: Verify OpenAI, Anthropic, and Gemini yield normalized StreamChunks."""
    gateway = UnifiedLLMGateway(
        openai_api_key="sk-test-openai-key",
        anthropic_api_key="sk-ant-test-anthropic-key",
        gemini_api_key="AIzaSyTestGeminiKey12345678901234567890",
    )

    # 1. Mock OpenAI
    openai_events = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Hello "))], usage=None),
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="OpenAI!"))], usage=None),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=10, completion_tokens=2)),
    ]
    gateway._openai_client.chat.completions.create = AsyncMock(
        return_value=AsyncIteratorMock(openai_events)
    )

    openai_chunks = [c async for c in gateway.stream("Say hi", model="gpt-4o-mini")]
    assert all(isinstance(c, StreamChunk) for c in openai_chunks)
    assert "".join(c.delta_text for c in openai_chunks) == "Hello OpenAI!"
    assert openai_chunks[-1].is_final is True

    # 2. Mock Anthropic
    anthropic_events = [
        SimpleNamespace(type="message_start", message=SimpleNamespace(usage=SimpleNamespace(input_tokens=12))),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Hello ")),
        SimpleNamespace(type="content_block_delta", delta=SimpleNamespace(text="Anthropic!")),
        SimpleNamespace(type="message_delta", usage=SimpleNamespace(output_tokens=3)),
    ]
    gateway._anthropic_client.messages.stream = MagicMock(
        return_value=AsyncContextManagerMock(anthropic_events)
    )

    anthropic_chunks = [c async for c in gateway.stream("Say hi", model="claude-3-5-haiku-latest")]
    assert all(isinstance(c, StreamChunk) for c in anthropic_chunks)
    assert "".join(c.delta_text for c in anthropic_chunks) == "Hello Anthropic!"
    assert anthropic_chunks[-1].is_final is True

    # 3. Mock Gemini
    gemini_events = [
        SimpleNamespace(text="Hello ", usage_metadata=None),
        SimpleNamespace(
            text="Gemini!",
            usage_metadata=SimpleNamespace(prompt_token_count=11, candidates_token_count=2),
        ),
    ]
    gateway._gemini_client.aio.models.generate_content_stream = AsyncMock(
        return_value=AsyncIteratorMock(gemini_events)
    )

    gemini_chunks = [c async for c in gateway.stream("Say hi", model="gemini-3.8-flash")]
    assert all(isinstance(c, StreamChunk) for c in gemini_chunks)
    assert "".join(c.delta_text for c in gemini_chunks) == "Hello Gemini!"
    assert gemini_chunks[-1].is_final is True


def test_token_count_validation():
    """Test 2: Verify deterministic prompt token calculation matches tiktoken within ±10%."""
    gateway = UnifiedLLMGateway(openai_api_key="sk-test")
    sample_prompt = " ".join(["architecture"] * 50)
    counted = gateway.count_tokens_local(sample_prompt, model="gpt-4o-mini")
    assert 45 <= counted <= 55


@pytest.mark.asyncio
async def test_telemetry_sanity():
    """Test 3: Assert final chunk telemetry has positive cost, TPS, and TTFT."""
    gateway = UnifiedLLMGateway(openai_api_key="sk-test")
    openai_events = [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content="Token"))], usage=None),
        SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=20, completion_tokens=5)),
    ]
    gateway._openai_client.chat.completions.create = AsyncMock(
        return_value=AsyncIteratorMock(openai_events)
    )

    chunks = [c async for c in gateway.stream("Test prompt", model="gpt-4o")]
    final_chunk = chunks[-1]

    assert final_chunk.is_final is True
    assert final_chunk.telemetry is not None
    assert final_chunk.telemetry.total_cost_usd > 0
    assert final_chunk.telemetry.tokens_per_second > 0
    assert final_chunk.telemetry.ttft_ms > 0