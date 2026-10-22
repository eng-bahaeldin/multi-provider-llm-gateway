from decimal import Decimal, ROUND_HALF_UP

# Pricing in USD per 1,000,000 tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    # Google Gemini Models
    "gemini-3.8-flash": {
        "input": 0.15,
        "output": 0.60,
    },
    "gemini-3.8-pro": {
        "input": 1.25,
        "output": 5.00,
    },
    # OpenAI Models
    "gpt-4o-mini": {
        "input": 0.15,
        "output": 0.60,
    },
    "gpt-4o": {
        "input": 2.50,
        "output": 10.00,
    },
    # Anthropic Models
    "claude-3-5-haiku-20241022": {
        "input": 0.80,
        "output": 4.00,
    },
    "claude-3-5-sonnet-20241022": {
        "input": 3.00,
        "output": 15.00,
    },
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Computes total execution cost down to micro-cents ($0.000001) using Decimal
    arithmetic to eliminate floating-point rounding drift.
    """
    rates = MODEL_PRICING.get(model, {"input": 0.0, "output": 0.0})
    million = Decimal("1000000")

    input_rate = Decimal(str(rates["input"]))
    output_rate = Decimal(str(rates["output"]))

    input_cost = (Decimal(prompt_tokens) / million) * input_rate
    output_cost = (Decimal(completion_tokens) / million) * output_rate
    total = input_cost + output_cost

    return float(total.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP))