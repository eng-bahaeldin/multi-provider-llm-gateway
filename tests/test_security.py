import logging
from unittest.mock import AsyncMock

import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from src.gateway import AuthenticationError, UnifiedLLMGateway


@pytest.mark.asyncio
async def test_missing_or_malformed_key_handling(monkeypatch):
    """Test 4: Missing or malformed keys raise typed AuthenticationError."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    gateway = UnifiedLLMGateway(
        openai_api_key="bad_openai_key",
        anthropic_api_key="",
        gemini_api_key="ya29.oauth_token_not_api_key",
    )

    for model_name in ("gpt-4o-mini", "claude-3-5-sonnet-latest", "gemini-3.8-flash"):
        with pytest.raises(AuthenticationError, match="missing or malformed"):
            async for _ in gateway.stream("Hello", model=model_name):
                pass


@pytest.mark.asyncio
async def test_secret_redaction_in_exceptions(caplog):
    """Test 5: Exception strings and logs must never expose raw API keys."""
    secret_openai_key = "sk-proj-SUPERSECRETKEY987654321"
    secret_gemini_key = "AIzaSySUPERSECRETGEMINIKEY1234567890123"

    gateway = UnifiedLLMGateway(
        openai_api_key=secret_openai_key,
        gemini_api_key=secret_gemini_key,
    )

    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_response = httpx.Response(status_code=401, request=mock_request)
    sdk_error = openai.AuthenticationError(
        message=f"Incorrect API key provided: {secret_openai_key}",
        response=mock_response,
        body=None,
    )
    gateway._openai_client.chat.completions.create = AsyncMock(side_effect=sdk_error)

    with caplog.at_level(logging.ERROR):
        with pytest.raises(AuthenticationError) as exc_info:
            async for _ in gateway.stream("Hello", model="gpt-4o-mini"):
                pass

    assert secret_openai_key not in str(exc_info.value)
    assert "[REDACTED_API_KEY]" in str(exc_info.value)
    assert secret_openai_key not in caplog.text

    # Also verify Gemini key redaction
    gemini_error = genai_errors.APIError(
        code=401,
        response_json={"error": {"message": f"Invalid key {secret_gemini_key}"}},
    )
    gateway._gemini_client.aio.models.generate_content_stream = AsyncMock(side_effect=gemini_error)

    with pytest.raises(AuthenticationError) as gemini_exc_info:
        async for _ in gateway.stream("Hello", model="gemini-3.8-flash"):
            pass

    assert secret_gemini_key not in str(gemini_exc_info.value)
    assert "[REDACTED_API_KEY]" in str(gemini_exc_info.value)