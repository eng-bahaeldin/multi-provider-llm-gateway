import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import inspect
import asyncio
import traceback
import streamlit as st

from src.gateway import GatewayError, UnifiedLLMGateway
from src.pricing import PRICING_TABLE
from src.schemas import ChunkTelemetry

st.set_page_config(page_title="Multi-Provider LLM Gateway", layout="wide")

st.title("Multi-Provider Async LLM Gateway & Telemetry Engine")
st.caption("Unified streaming abstraction across OpenAI, Anthropic, and Google Gemini")


with st.sidebar:
    st.header("Model Configuration")
    model_options = list(PRICING_TABLE.keys())
    model = st.selectbox("Select Model", options=model_options, index=0)
    provider = PRICING_TABLE[model]["provider"].upper()
    st.info(f"**Active Provider:** {provider}")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)

    if st.button("Reload .env Credentials"):
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.header("Live Telemetry")
    ttft_metric = st.empty()
    tps_metric = st.empty()
    tokens_metric = st.empty()
    cost_metric = st.empty()

    ttft_metric.metric("TTFT (Time to First Token)", "0.00 ms")
    tps_metric.metric("Throughput", "0.00 tok/s")
    tokens_metric.metric("Tokens (In / Out)", "0 / 0")
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


        async def run_stream() -> None:
            # Create a fresh gateway inside the running event loop
            local_gateway = UnifiedLLMGateway()

            accumulated_text = ""
            final_telemetry: ChunkTelemetry | None = None

            try:
                async for chunk in local_gateway.stream(
                        prompt=prompt,
                        model=model,
                        temperature=temperature
                ):
                    if chunk.delta_text:
                        accumulated_text += chunk.delta_text
                        output_box.markdown(accumulated_text + "▌")

                    if chunk.is_final and chunk.telemetry:
                        final_telemetry = chunk.telemetry

                output_box.markdown(accumulated_text)

                if final_telemetry:
                    ttft_metric.metric(
                        "TTFT (Time to First Token)",
                        f"{final_telemetry.ttft_ms:.2f} ms"
                    )

                    tps_metric.metric(
                        "Throughput",
                        f"{final_telemetry.tokens_per_second:.2f} tok/s"
                    )

                    tokens_metric.metric(
                        "Tokens (In / Out)",
                        f"{final_telemetry.input_tokens} / "
                        f"{final_telemetry.output_tokens}"
                    )

                    cost_metric.metric(
                        "Execution Cost",
                        f"${final_telemetry.total_cost_usd:.6f}"
                    )

            finally:
                # Close asynchronous resources if the gateway supports cleanup.
                close_method = (
                        getattr(local_gateway, "aclose", None)
                        or getattr(local_gateway, "close", None)
                )

                if callable(close_method):
                    result = close_method()

                    if inspect.isawaitable(result):
                        await result

        try:
            asyncio.run(run_stream())
        except GatewayError as err:
            output_box.empty()
            st.error(f"Gateway Error: {err}")
        except Exception as err:
            output_box.empty()
            traceback.print_exc()
            st.error(f"Internal Error ({type(err).__name__}): {err}")