"""src/pricing.py - Model pricing tables and deterministic micro-cent cost calculation."""

from typing import Dict, Tuple

PRICING_TABLE_PER_1M: Dict[str, Tuple[float, float]] = {
    # Google Gemini
    "gemini-3.8-flash": (0.10, 0.40),
    "gemini-3.8-pro": (1.25, 5.00),
    "gemini-3.5-flash": (0.10, 0.40),
    "gemini-3.5-pro": (1.25, 5.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (1.00, 5.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-latest": (1.00, 5.00),
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculates execution cost in USD down to micro-cents (6 decimal places)."""
    input_rate, output_rate = PRICING_TABLE_PER_1M.get(model, (0.15, 0.60))
    input_cost = (prompt_tokens / 1_000_000.0) * input_rate
    output_cost = (completion_tokens / 1_000_000.0) * output_rate
    raw_cost = input_cost + output_cost

    if raw_cost > 0.0:
        return max(round(raw_cost, 6), 0.000001)
    return 0.0
