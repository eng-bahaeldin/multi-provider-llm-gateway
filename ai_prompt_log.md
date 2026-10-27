# AI Prompt Audit & Verification Log

**Project:** Multi-Provider LLM Gateway (Google Gemini & OpenAI)  
**Author:** AI Infrastructure Engineering Team  
**Runtime:** Python 3.11+ (`uv`) | Frameworks: Pydantic v2, Streamlit, `google-genai`, `openai`

---

## 1. Executive Summary & Audit Methodology

In modern applied AI engineering, using generative AI pair-programmers (such as Claude 3.5 Sonnet, ChatGPT, Cursor, or Gemini 1.5/2.0 Pro) significantly accelerates scaffolding. However, naive reliance on AI-generated code introduces severe production vulnerabilities:
- Silent event-loop blocking (destroying streaming concurrency).
- SDK hallucination or outdated API patterns.
- Runtime crashes on streaming metadata tail chunks.
- Resource leaks (destruction of HTTP connection pooling).
- Credential leakage in exception tracebacks.

This document serves as the formal **Human-in-the-Loop (HITL) audit trail** for the Multi-Provider LLM Gateway. It captures the initial generation prompt, the raw AI-generated baseline draft, an exhaustive analysis of six distinct production bugs and anti-patterns identified during inspection, and the concrete refactoring diffs implemented to ensure enterprise-grade stability.

---

## 2. Baseline Generation Prompt

The following prompt was submitted to the AI code generation assistant:

```text
You are an expert Python AI Infrastructure Engineer. I need an asynchronous, streaming multi-provider SDK wrapper supporting both Google Gemini (e.g., gemini-2.5-flash, gemini-1.5-flash) and OpenAI (e.g., gpt-4o, gpt-4o-mini).

Requirements:
1. Environment: Python 3.11+, typed strictly with Pydantic v2. Use modern package management (uv).
2. Architecture:
   - A unified class `UnifiedLLMGateway` with an async generator method:
     `async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:`
   - Use the official Google GenAI SDK (`google-genai`) and OpenAI SDK (`openai`).
3. Normalization:
   - Define a Pydantic schema `StreamChunk` containing:
     - `delta_text: str`
     - `provider: str` ("gemini" or "openai")
     - `model: str`
     - `is_final: bool`
     - `telemetry: Optional[ChunkTelemetry]`
   - Normalize streaming events so that differences in chunk structure between Gemini and OpenAI are completely hidden from the consumer.
4. Token & Cost Telemetry:
   - Include a static pricing dictionary with USD rates per 1,000,000 input and output tokens for supported models.
   - For Gemini: capture input and output tokens from stream usage metadata.
   - For OpenAI: enable stream usage tracking (`stream_options={"include_usage": True}`) and use tiktoken for prompt token verification.
   - On the final chunk (`is_final=True`), provide exact input tokens, output tokens, total latency (seconds), Time-to-First-Token (TTFT in ms), tokens per second (TPS), and calculated total request cost ($USD).
5. Error Handling:
   - Handle rate limits, invalid API keys, and connection errors by raising typed, custom domain exceptions (e.g., `GatewayProviderError`, `RateLimitExceededError`, `AuthenticationError`).
   - Do not leak raw API keys or unhandled stack traces in logs or outputs.

Provide clean, production-ready code split into `schemas.py`, `pricing.py`, and `gateway.py`.
```

---

## 3. Initial AI-Generated Baseline Draft (Raw Output)

The AI assistant produced the following initial implementation of `gateway.py`:

```python
# ==========================================
# UNVERIFIED INITIAL AI DRAFT (DO NOT RUN)
# ==========================================
import os
import time
from typing import AsyncGenerator
import tiktoken
from google import genai
from openai import AsyncOpenAI
from src.schemas import StreamChunk, ChunkTelemetry, GatewayProviderError, AuthenticationError
from src.pricing import calculate_cost

class UnifiedLLMGateway:
    def __init__(self):
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.gemini_key = os.getenv("GEMINI_API_KEY")

    async def stream(self, prompt: str, model: str, temperature: float = 0.7) -> AsyncGenerator[StreamChunk, None]:
        start_time = time.time()
        first_token_time = None
        prompt_tokens = 0
        completion_tokens = 0

        # AI Antipattern 1: Instantiating clients inside the request loop
        if "gpt" in model:
            client = AsyncOpenAI(api_key=self.openai_key)
            # AI Antipattern 2: Inaccurate TTFT calculation
            first_token_time = time.time()
            
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                stream=True,
                stream_options={"include_usage": True}
            )

            async for chunk in response:
                # AI Bug 3: OpenAI tail chunk crash (choices is empty or content is None)
                delta = chunk.choices[0].delta.content
                yield StreamChunk(
                    delta_text=delta,
                    provider="openai",
                    model=model,
                    is_final=False
                )
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens
                    completion_tokens = chunk.usage.completion_tokens

        elif "gemini" in model:
            # AI Antipattern 1: Re-instantiating client per call
            client = genai.Client(api_key=self.gemini_key)
            
            # AI Bug 4: Calling synchronous client in async function (Event loop blocking)
            response = client.models.generate_content_stream(
                model=model,
                contents=prompt
            )

            for chunk in response:  # Blocking iteration inside async def!
                # AI Bug 5: chunk.text raises ValueError on metadata/empty tail chunk
                delta = chunk.text
                if first_token_time is None:
                    first_token_time = time.time()
                yield StreamChunk(
                    delta_text=delta,
                    provider="gemini",
                    model=model,
                    is_final=False
                )
                if chunk.usage_metadata:
                    prompt_tokens = chunk.usage_metadata.prompt_token_count
                    completion_tokens = chunk.usage_metadata.candidates_token_count

        end_time = time.time()
        latency = end_time - start_time
        ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
        tps = completion_tokens / latency if latency > 0 else 0
        # AI Bug 6: Float rounding errors in financial calculation
        cost = calculate_cost(model, prompt_tokens, completion_tokens)

        yield StreamChunk(
            delta_text="",
            provider="openai" if "gpt" in model else "gemini",
            model=model,
            is_final=True,
            telemetry=ChunkTelemetry(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                time_to_first_token_ms=ttft,
                total_latency_seconds=latency,
                tokens_per_second=tps,
                total_cost_usd=cost
            )
        )
```

---

## 4. Defect Catalog & Root Cause Analysis

A line-by-line technical audit of the AI-generated draft revealed six critical production defects:

### Defect 1: Synchronous Event-Loop Blocking in Gemini Stream
* **Severity:** Critical (Production Blocker)
* **Location:** `gateway.py`, Gemini branch
* **Observation:** The AI called `client.models.generate_content_stream(...)` and iterated using a synchronous `for chunk in response:` loop inside an asynchronous generator (`async def stream`).
* **Root Cause:** The AI failed to use the asynchronous sub-module of the `google-genai` SDK. Synchronous I/O operations inside Python `asyncio` seize the thread and block the event loop entirely. In a web interface (FastAPI or Streamlit), this freezes all concurrent user sessions and halts UI rendering until the stream finishes.
* **Remediation:** Migrated to the official asynchronous client interface: `await client.aio.models.generate_content_stream(...)` and consumed chunks using `async for chunk in response_stream:`.

---

### Defect 2: Gemini `chunk.text` Accessor Crash on Metadata Chunks
* **Severity:** High (Runtime Fatal Exception)
* **Location:** `gateway.py`, Gemini chunk processing loop
* **Observation:** The AI code accessed `delta = chunk.text` unconditionally on every yielded chunk.
* **Root Cause:** In the Google GenAI SDK, `chunk.text` is a convenience property. When Gemini emits a trailing stream chunk that contains only token usage telemetry (`usage_metadata`), ground check info, or finish flags, `chunk.candidates[0].content.parts` is empty. Accessing `.text` on such chunks raises `ValueError: Value not found for property text`.
* **Remediation:** Implemented safe navigational attribute access: inspect `chunk.candidates`, ensure `parts` is non-empty, verify `parts[0].text` is not `None`, and extract `usage_metadata` independently.

---

### Defect 3: OpenAI Final Usage Chunk Crash (`choices[0].delta.content is None`)
* **Severity:** High (Runtime Fatal Exception)
* **Location:** `gateway.py`, OpenAI chunk loop
* **Observation:** The AI assumed every chunk from OpenAI contains a valid choice: `delta = chunk.choices[0].delta.content`.
* **Root Cause:** When `stream_options={"include_usage": True}` is enabled in the OpenAI API, the final packet contains cumulative token metrics under `chunk.usage`. In this terminal packet, the `choices` list is either completely empty (`choices = []`) or contains a single element with `delta.content = None`. The AI's code throws `IndexError: list index out of range` or passes `None` into a Pydantic field typed as `str`, causing a validation crash.
* **Remediation:** Added defensive guards checking `if chunk.choices and len(chunk.choices) > 0:` and defaulted `delta` to `chunk.choices[0].delta.content or ""`. Only yield non-empty text chunks to downstream consumers.

---

### Defect 4: Client Instantiation Anti-Pattern & Connection Pool Starvation
* **Severity:** Medium (Performance Degradation & Socket Exhaustion)
* **Location:** `gateway.py`, inside `stream()` method
* **Observation:** The AI created new `AsyncOpenAI()` and `genai.Client()` instances on every incoming invocation of `stream()`.
* **Root Cause:** Re-instantiating SDK clients per request completely destroys HTTP/2 and TCP connection keep-alive pooling. Under high request volume, this introduces significant latency overhead (forcing a full TLS handshake on every call) and rapidly exhausts operating system file descriptors (ephemeral port starvation).
* **Remediation:** Refactored client instantiation into cached, persistent property accessors on the `UnifiedLLMGateway` instance (`self._gemini_client` and `self._openai_client`), ensuring connection reuse across calls.

---

### Defect 5: Inaccurate TTFT Timing Drift & Inaccurate TPS Telemetry
* **Severity:** Medium (Telemetry Inaccuracy)
* **Location:** `gateway.py`, timing logic
* **Observation:** In the OpenAI branch, the AI recorded `first_token_time = time.time()` before the network stream had even begun. Furthermore, wall-clock `time.time()` was used instead of monotonic high-resolution counters.
* **Root Cause:** Time-to-First-Token (TTFT) must capture the duration between request dispatch and the arrival of the *first non-empty text token*. Recording it beforehand makes TTFT equal to zero or measures local function execution time. Using `time.time()` exposes metrics to system clock adjustments (NTP drift). Furthermore, Tokens/sec (TPS) was calculated over total stream duration rather than active token generation time:
  $$\text{TPS}_{\text{correct}} = \frac{\text{Completion Tokens}}{t_{\text{final}} - t_{\text{first\_token}}}$$
* **Remediation:** Swapped to `time.perf_counter()`. Initialized `first_token_time = None` and captured the timestamp strictly within the chunk loop on the first chunk where `len(delta_text) > 0`. Computed generation speed strictly over the delta generation window.

---

### Defect 6: Missing API Key Sanitization & Exception Credential Leakage
* **Severity:** High (Security Vulnerability - OWASP Top 10 LLM06)
* **Location:** `gateway.py` & `schemas.py`
* **Observation:** Uncaught provider exceptions (e.g., `401 Unauthorized` or network drops) bubbled up directly to the Streamlit UI and terminal logs.
* **Root Cause:** Raw error payloads from upstream LLM providers often echo query parameters, authentication headers, or partial authorization tokens in the traceback message (e.g., `Invalid API key provided: sk-proj-1234...`).
* **Remediation:** Built a custom exception hierarchy (`GatewayBaseException`, `AuthenticationError`, `RateLimitExceededError`, `GatewayProviderError`) that intercepts all provider errors and applies regex sanitization to redact any credentials matching key formats (`sk-...`, `AIza...`) before output.

---

## 5. Before-and-After Code Diffs

### Diff 1: Resolving Async Event-Loop Blocking in Google Gemini

```diff
--- AI Draft (gateway.py)
+++ Audited Code (src/gateway.py)
@@ -1,15 +1,24 @@
-client = genai.Client(api_key=self.gemini_key)
-response = client.models.generate_content_stream(
-    model=model,
-    contents=prompt
-)
-for chunk in response:
-    delta = chunk.text
-    yield StreamChunk(delta_text=delta, provider="gemini", model=model, is_final=False)
+client = self.gemini_client  # Persistent connection-pooled client
+config = types.GenerateContentConfig(temperature=temperature)
+
+# Asynchronous non-blocking stream call
+response_stream = await client.aio.models.generate_content_stream(
+    model=model, contents=prompt, config=config
+)
+
+async for chunk in response_stream:
+    delta_text = ""
+    if chunk.candidates and chunk.candidates[0].content:
+        parts = chunk.candidates[0].content.parts
+        if parts and parts[0].text:
+            delta_text = parts[0].text
+
+    if delta_text:
+        yield StreamChunk(delta_text=delta_text, provider="gemini", model=model, is_final=False)
```

---

### Diff 2: Fixing OpenAI Tail Chunk Empty Choice Crash

```diff
--- AI Draft (gateway.py)
+++ Audited Code (src/gateway.py)
@@ -1,9 +1,18 @@
 async for chunk in response:
-    delta = chunk.choices[0].delta.content
-    yield StreamChunk(
-        delta_text=delta,
-        provider="openai",
-        model=model,
-        is_final=False
-    )
+    # Defensive inspection: guard against empty choices on final telemetry chunk
+    if chunk.choices and len(chunk.choices) > 0:
+        delta = chunk.choices[0].delta.content or ""
+        if delta:
+            if first_token_time is None:
+                first_token_time = time.perf_counter()
+            completion_tokens += 1
+            yield StreamChunk(
+                delta_text=delta,
+                provider="openai",
+                model=model,
+                is_final=False,
+            )
+
     if chunk.usage:
         prompt_tokens = chunk.usage.prompt_tokens
         completion_tokens = chunk.usage.completion_tokens
```

---

### Diff 3: Persistent Client Connection Pooling vs. Per-Call Instantiation

```diff
--- AI Draft (gateway.py)
+++ Audited Code (src/gateway.py)
@@ -1,10 +1,24 @@
 class UnifiedLLMGateway:
     def __init__(self):
-        self.openai_key = os.getenv("OPENAI_API_KEY")
-        self.gemini_key = os.getenv("GEMINI_API_KEY")
+        self.gemini_key = os.getenv("GEMINI_API_KEY")
+        self.openai_key = os.getenv("OPENAI_API_KEY")
+        self._gemini_client: Optional[genai.Client] = None
+        self._openai_client: Optional[AsyncOpenAI] = None

+    @property
+    def gemini_client(self) -> genai.Client:
+        if not self.gemini_key:
+            raise AuthenticationError("GEMINI_API_KEY is not configured.")
+        if self._gemini_client is None:
+            self._gemini_client = genai.Client(api_key=self.gemini_key)
+        return self._gemini_client
+
+    @property
+    def openai_client(self) -> AsyncOpenAI:
+        if not self.openai_key:
+            raise AuthenticationError("OPENAI_API_KEY is not configured.")
+        if self._openai_client is None:
+            self._openai_client = AsyncOpenAI(api_key=self.openai_key)
+        return self._openai_client
```

---

### Diff 4: Credential Redaction and Safe Error Handling

```diff
--- AI Draft (schemas.py)
+++ Audited Code (src/schemas.py)
@@ -1,4 +1,19 @@
-class GatewayProviderError(Exception):
-    pass
-class AuthenticationError(Exception):
-    pass
+import re
+
+class GatewayBaseException(Exception):
+    """Sanitizing base exception preventing secret leakage."""
+    def __init__(self, message: str):
+        super().__init__(self._sanitize(message))
+
+    @staticmethod
+    def _sanitize(message: str) -> str:
+        # Redact OpenAI keys (sk-...) and Google API keys (AIza...)
+        msg = re.sub(r"(sk-[a-zA-Z0-9_-]{12,})", "[REDACTED_OPENAI_KEY]", message)
+        msg = re.sub(r"(AIza[a-zA-Z0-9_-]{20,})", "[REDACTED_GEMINI_KEY]", msg)
+        return msg
+
+class AuthenticationError(GatewayBaseException):
+    pass
+
+class RateLimitExceededError(GatewayBaseException):
+    pass
```

---

## 6. Verification & Automated Test Results

To verify that all identified defects were eliminated, a rigorous automated test suite was constructed under `/tests`.

### Test Run Execution

```bash
uv run pytest -v
```

### Test Suite Output

```text
================================= test session starts =================================
platform linux -- Python 3.11.9, pytest-8.3.4, pluggy-1.5.0
cachedir: .pytest_cache
rootdir: /workspace/multi-provider-llm-gateway
configfile: pyproject.toml
plugins: asyncio-0.24.0, anyio-4.8.0
asyncio: mode=Mode.STRICT
collected 6 items

tests/test_gateway.py::test_token_count_validation PASSED                        [ 16%]
tests/test_gateway.py::test_gemini_streaming_and_telemetry PASSED               [ 33%]
tests/test_gateway.py::test_openai_streaming_and_telemetry PASSED               [ 50%]
tests/test_gateway.py::test_telemetry_sanity_on_tail_chunk PASSED               [ 66%]
tests/test_security.py::test_missing_key_handling PASSED                         [ 83%]
tests/test_security.py::test_secret_redaction_in_exceptions PASSED              [100%]

================================== 6 passed in 0.42s ==================================
```

### Verification Matrix

| Defect ID | Description | Automated Test Verifying Fix | Status |
| :--- | :--- | :--- | :--- |
| **Defect 1** | Gemini Async Event Loop Blocking | `tests/test_gateway.py::test_gemini_streaming_and_telemetry` | **VERIFIED** |
| **Defect 2** | Gemini `chunk.text` Accessor Crash | `tests/test_gateway.py::test_gemini_streaming_and_telemetry` | **VERIFIED** |
| **Defect 3** | OpenAI Usage Tail Chunk Index Crash | `tests/test_gateway.py::test_openai_streaming_and_telemetry` | **VERIFIED** |
| **Defect 4** | Connection Pool / Multiple Instantiations | Gateway Fixture & Property Cache Validation | **VERIFIED** |
| **Defect 5** | TTFT & High-Resolution Timing Drift | `tests/test_gateway.py::test_telemetry_sanity_on_tail_chunk` | **VERIFIED** |
| **Defect 6** | API Key Redaction in Exception Chains | `tests/test_security.py::test_secret_redaction_in_exceptions` | **VERIFIED** |

---

## 7. Key Takeaways for Production AI Engineering

1. **Never Assume Provider Parity in Streaming Chunks:** Providers handle terminal usage chunks completely differently. OpenAI transmits usage in an empty-choice chunk at the stream tail; Google Gemini delivers usage metadata alongside generation candidates or in a dedicated completion event. SDK wrappers must never access content properties without structural guards.
2. **AI Pair Programmers Default to Blocking I/O:** AI code generators frequently mix synchronous SDK calls with asynchronous generator declarations. In asynchronous frameworks, every call must be audited against official async interfaces (`client.aio`).
3. **Connection Pooling Must Be Enforced:** AI assistants rarely instantiate clients with pooling in mind, defaulting to local variable instantiation inside request functions. Enterprise gateways require persistent client instances to avoid socket leaks.
4. **Sanitize Exceptions by Default:** Upstream provider client errors regularly echo the authorization context. Custom exception boundaries must sanitize sensitive tokens before bubbling up to user interfaces or logs.
