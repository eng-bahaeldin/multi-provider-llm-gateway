import sys
from pathlib import Path
import sys
from pathlib import Path

# Force the project root directory onto sys.path before local imports
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import asyncio
import streamlit as st
from dotenv import load_dotenv

from src.gateway import UnifiedLLMGateway
from src.schemas import GatewayError

# Add project root directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import asyncio
import streamlit as st
from dotenv import load_dotenv
from src.gateway import UnifiedLLMGateway
from src.schemas import GatewayError
import asyncio
import streamlit as st
from dotenv import load_dotenv
from src.gateway import UnifiedLLMGateway
from src.schemas import GatewayError

load_dotenv()

st.set_page_config(
    page_title="Multi-Provider LLM Gateway",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Multi-Provider LLM Gateway")
st.caption("Normalized async streaming with micro-cent cost accounting.")

# ==============================================================================
# Sidebar Controls
# ==============================================================================
st.sidebar.header("Model Configuration")

model_directory = {
    "Google Gemini": ["gemini-3.8-flash", "gemini-3.8-pro"],
    "OpenAI": ["gpt-4o-mini", "gpt-4o"],
    "Anthropic": ["claude-3-5-haiku-20241022", "claude-3-5-sonnet-20241022"],
}

provider_choice = st.sidebar.radio("Provider", list(model_directory.keys()))
selected_model = st.sidebar.selectbox("Model", model_directory[provider_choice])
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.05)

# Sidebar Live Telemetry Widgets
st.sidebar.markdown("---")
st.sidebar.subheader("📊 Execution Telemetry")
metric_ttft = st.sidebar.empty()
metric_tps = st.sidebar.empty()
metric_tokens = st.sidebar.empty()
metric_cost = st.sidebar.empty()

# Default metric placeholders
metric_ttft.metric("Time to First Token (TTFT)", "-- ms")
metric_tps.metric("Throughput Speed", "-- tok/s")
metric_tokens.metric("Tokens (Prompt / Output)", "-- / --")
metric_cost.metric("Execution Cost ($USD)", "$0.000000")

# ==============================================================================
# Main Workspace
# ==============================================================================
user_prompt = st.text_area(
    "Enter Prompt:",
    value="Explain why software architecture should decouple application logic from foundation model SDKs in three sentences.",
    height=120
)
submit_btn = st.button("Generate Stream", type="primary")


async def run_streaming_job(prompt_str: str, model_str: str, temp_val: float):
    gateway = UnifiedLLMGateway()
    output_container = st.empty()
    accumulated_text = ""

    try:
        async for chunk in gateway.stream(prompt=prompt_str, model=model_str, temperature=temp_val):
            if not chunk.is_final:
                accumulated_text += chunk.delta_text
                # Streaming cursor effect
                output_container.markdown(accumulated_text + " ▌")
            else:
                # Remove cursor on terminal chunk
                output_container.markdown(accumulated_text)

                # Render final telemetry receipt
                if chunk.telemetry:
                    t = chunk.telemetry
                    metric_ttft.metric("Time to First Token (TTFT)", f"{t.ttft_ms:.1f} ms")
                    metric_tps.metric("Throughput Speed", f"{t.tokens_per_second:.1f} tok/s")
                    metric_tokens.metric("Tokens (Prompt / Output)", f"{t.prompt_tokens} / {t.completion_tokens}")
                    metric_cost.metric("Execution Cost ($USD)", f"${t.total_cost_usd:.6f}")

    except GatewayError as e:
        st.error(f"❌ {e}")
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")


if submit_btn:
    if not user_prompt.strip():
        st.warning("Please enter a non-empty prompt.")
    else:
        asyncio.run(run_streaming_job(user_prompt, selected_model, temperature))