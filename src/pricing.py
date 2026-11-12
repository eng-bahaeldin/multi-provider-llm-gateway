from decimal import Decimal, ROUND_HALF_UP
from typing import Literal, TypedDict


class ModelPricing(TypedDict):
    provider: Literal["openai", "anthropic", "gemini"]
    input_per_million: float
    output_per_million: float


MODEL_PRICING: dict[str, dict[str, float]] = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-flash-lite": {"input": 0.25, "output": 1.50},
    "gemini-3.8-flash": {"input": 0.15, "output": 0.60},
    "gemini-3.8-pro": {"input": 1.25, "output": 5.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "claude-3-5-haiku-latest": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-latest": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
}


PRICING_TABLE: dict[str, ModelPricing] = {
    "gemini-3.5-flash-lite": {"provider": "gemini", "input_per_million": 0.30, "output_per_million": 2.50},
    "gemini-3.1-flash-lite": {"provider": "gemini", "input_per_million": 0.25, "output_per_million": 1.50},
    "gemini-3.8-flash": {"provider": "gemini", "input_per_million": 0.15, "output_per_million": 0.60},
    "gemini-3.8-pro": {"provider": "gemini", "input_per_million": 1.25, "output_per_million": 5.00},
    "gpt-4o-mini": {"provider": "openai", "input_per_million": 0.15, "output_per_million": 0.60},
    "gpt-4o": {"provider": "openai", "input_per_million": 2.50, "output_per_million": 10.00},
    "claude-3-5-haiku-latest": {"provider": "anthropic", "input_per_million": 0.80, "output_per_million": 4.00},
    "claude-3-5-sonnet-latest": {"provider": "anthropic", "input_per_million": 3.00, "output_per_million": 15.00},
    "claude-3-5-haiku-20241022": {"provider": "anthropic", "input_per_million": 0.80, "output_per_million": 4.00},
    "claude-3-5-sonnet-20241022": {"provider": "anthropic", "input_per_million": 3.00, "output_per_million": 15.00},
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Compute the total execution cost in USD using Decimal arithmetic."""
    rates = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    million = Decimal("1000000")

    input_rate = Decimal(str(rates["input"]))
    output_rate = Decimal(str(rates["output"]))

    input_cost = (Decimal(prompt_tokens) / million) * input_rate
    output_cost = (Decimal(completion_tokens) / million) * output_rate
    total = input_cost + output_cost

    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))


def calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Compatibility wrapper for the newer input/output token naming scheme."""
    pricing = PRICING_TABLE.get(model)
    if pricing is None:
        return calculate_cost(model, input_tokens, output_tokens)

    input_cost = (Decimal(input_tokens) / Decimal("1000000")) * Decimal(str(pricing["input_per_million"]))
    output_cost = (Decimal(output_tokens) / Decimal("1000000")) * Decimal(str(pricing["output_per_million"]))
    total = input_cost + output_cost
    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))
