"""src/app.py - Streamlit web interface with throttled live streaming and cost accounting."""

import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asyncio
import queue
import threading
import time
import traceback
from typing import Any, Optional

from dotenv import load_dotenv
import streamlit as st

from src.gateway import GatewayError, UnifiedLLMGateway, sanitize_message
from src.pricing import PRICING_TABLE_PER_1M
from src.schemas import ChunkTelemetry

load_dotenv(".env", override=True)

# Throttle WebSocket UI repaints to ~28 FPS to prevent browser/network lag during fast streams
_UI_RENDER_INTERVAL_SEC = 0.035


def _sync_streamlit_secrets() -> None:
    """Mirrors Streamlit Cloud secrets into environment variables if present."""
    try:
        for key in ("GEMINI_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = str(st.secrets[key])
    except Exception:
        pass


_sync_streamlit_secrets()

st.set_page_config(page_title="Multi-Provider LLM Gateway", layout="wide")

st.title("Multi-Provider Async LLM Gateway & Telemetry Engine")
st.caption("Unified streaming abstraction across OpenAI, Anthropic, and Google Gemini")


class AsyncEventLoopRunner:
    """Maintains a persistent, self-healing SelectorEventLoop on a background daemon thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._start_loop()

    def _start_loop(self) -> None:
        ready = threading.Event()
        if sys.platform == "win32":
            loop = asyncio.SelectorEventLoop()
        else:
            loop = asyncio.new_event_loop()

        def _runner() -> None:
            asyncio.set_event_loop(loop)
            loop.call_soon(ready.set)
            loop.run_forever()

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        ready.wait(timeout=5.0)
        self.loop = loop
        self._thread = thread

    def submit(self, coro_factory: Any) -> None:
        """Submits a coroutine factory to the active loop, restarting the loop if closed."""
        with self._lock:
            if (
                self.loop is None
                or self.loop.is_closed()
                or not self.loop.is_running()
                or self._thread is None
                or not self._thread.is_alive()
            ):
                self._start_loop()
            assert self.loop is not None
            try:
                asyncio.run_coroutine_threadsafe(coro_factory(), self.loop)
            except RuntimeError:
                self._start_loop()
                assert self.loop is not None
                asyncio.run_coroutine_threadsafe(coro_factory(), self.loop)


@st.cache_resource
def get_loop_runner() -> AsyncEventLoopRunner:
    """Returns a process-wide background event loop runner across Streamlit reruns."""
    return AsyncEventLoopRunner()


@st.cache_resource
def get_gateway() -> UnifiedLLMGateway:
    """Shares a single connection-pooled gateway instance across reruns."""
    return UnifiedLLMGateway()


def _detect_provider_label(model_name: str) -> str:
    lower = model_name.lower()
    if "gemini" in lower:
        return "GEMINI"
    if "claude" in lower:
        return "ANTHROPIC"
    return "OPENAI"


gateway = get_gateway()

with st.sidebar:
    st.header("Model Configuration")
    model_options = list(PRICING_TABLE_PER_1M.keys())
    default_idx = model_options.index("gemini-3.8-flash") if "gemini-3.8-flash" in model_options else 0
    model = st.selectbox("Select Model", options=model_options, index=default_idx)
    provider = _detect_provider_label(model)
    st.info(f"**Active Provider:** {provider}")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    if st.button("Reload .env Credentials"):
        load_dotenv(".env", override=True)
        _sync_streamlit_secrets()
        get_gateway.clear()
        st.rerun()

    st.divider()
    st.header("Live Telemetry")
    ttft_metric = st.empty()
    tps_metric = st.empty()
    tokens_metric = st.empty()
    cost_metric = st.empty()

    ttft_metric.metric("TTFT (Time to First Token)", "0.00 ms")
    tps_metric.metric("Throughput", "0.00 tok/s")
    tokens_metric.metric("Tokens (In / Out / Total)", "0 / 0 / 0")
    cost_metric.metric("Execution Cost", "$0.000000")

prompt = st.text_area(
    "Enter your prompt:",
    value="Explain how asynchronous event loops prevent I/O blocking in 3 concise bullet points.",
    height=120,
)

if st.button("Stream Response", type="primary"):
    if not prompt.strip():
        st.warning("Please enter a prompt before submitting.")
    else:
        output_box = st.empty()
        output_box.info(f"Connecting to {provider} (`{model}`)...")

        runner = get_loop_runner()
        chunk_queue: queue.Queue[tuple[str, Any, str]] = queue.Queue()

        async def _produce_stream() -> None:
            try:
                async for chunk in gateway.stream(
                    prompt=prompt,
                    model=model,
                    temperature=temperature,
                ):
                    chunk_queue.put(("chunk", chunk, ""))
            except Exception as exc:
                tb_lines = traceback.format_exc().strip().splitlines()
                location = " | ".join(tb_lines[-3:]) if tb_lines else ""
                chunk_queue.put(("error", exc, sanitize_message(location)))
            finally:
                chunk_queue.put(("done", None, ""))

        runner.submit(_produce_stream)

        accumulated_text = ""
        final_telemetry: Optional[ChunkTelemetry] = None
        stream_failed = False
        last_render_time = 0.0

        while True:
            kind, payload, location = chunk_queue.get()
            if kind == "chunk":
                if payload.delta_text:
                    accumulated_text += payload.delta_text
                    now = time.perf_counter()
                    if now - last_render_time >= _UI_RENDER_INTERVAL_SEC:
                        output_box.markdown(accumulated_text + "▌")
                        last_render_time = now
                if payload.is_final and payload.telemetry:
                    final_telemetry = payload.telemetry
            elif kind == "error":
                stream_failed = True
                output_box.empty()
                if isinstance(payload, GatewayError):
                    st.error(f"Gateway Error: {payload}")
                else:
                    st.error(
                        f"Internal Error ({type(payload).__name__}): {sanitize_message(payload)}\n\n"
                        f"`{location}`"
                    )
            elif kind == "done":
                break

        if not stream_failed:
            output_box.markdown(accumulated_text)

        if final_telemetry:
            ttft_metric.metric(
                "TTFT (Time to First Token)",
                f"{final_telemetry.time_to_first_token_ms:.2f} ms",
            )
            tps_metric.metric("Throughput", f"{final_telemetry.tokens_per_second:.2f} tok/s")
            tokens_metric.metric(
                "Tokens (In / Out / Total)",
                f"{final_telemetry.prompt_tokens} / {final_telemetry.completion_tokens} / {final_telemetry.total_tokens}",
            )
            cost_metric.metric("Execution Cost", f"${final_telemetry.total_cost_usd:.6f}")
