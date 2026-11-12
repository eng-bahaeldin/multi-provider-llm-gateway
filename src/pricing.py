from typing import Literal, TypedDict


class ModelPricing(TypedDict):
    provider: Literal["openai", "anthropic", "gemini"]
    input_per_million: float
    output_per_million: float


# Pricing in USD per 1,000,000 tokens
PRICING_TABLE: dict[str, ModelPricing] = {
    "gemini-3.5-flash-lite": {
        "provider": "gemini",
        "input_per_million": 0.30,
        "output_per_million": 2.50,
    },
    "gemini-3.1-flash-lite": {
        "provider": "gemini",
        "input_per_million": 0.25,
        "output_per_million": 1.50,
    },
    "gemini-3.8-flash": {
        "provider": "gemini",
        "input_per_million": 0.75,
        "output_per_million": 3.75,
    },
    "gpt-4o-mini": {
        "provider": "openai",
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    "gpt-4o": {
        "provider": "openai",
        "input_per_million": 2.50,
        "output_per_million": 10.00,
    },
    "claude-3-5-sonnet-latest": {
        "provider": "anthropic",
        "input_per_million": 3.00,
        "output_per_million": 15.00,
    },
    "claude-3-5-haiku-latest": {
        "provider": "anthropic",
        "input_per_million": 0.80,
        "output_per_million": 4.00,
    },
}


def calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculates execution cost in USD rounded to 6 decimal places (micro-cents)."""
    pricing = PRICING_TABLE.get(model)
    if not pricing:
        raise ValueError(f"Unsupported model for pricing calculation: {model}")

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]
    return round(input_cost + output_cost, 6)