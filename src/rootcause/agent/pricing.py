"""Token pricing (USD per million tokens) for cost tracking. Update as prices change."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float


PRICES: dict[str, Price] = {
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_read=0.20, cache_write=2.50),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_read=0.20, cache_write=5.00),
    "claude-haiku-5-5": Price(input=0.10, output=0.50, cache_read=0.01, cache_write=0.125),
    "claude-fable-5-1": Price(input=10.00, output=50.00, cache_read=0.25, cache_write=12.50),
    "fake-rule-based": Price(input=0.0, output=0.0, cache_read=0.0, cache_write=0.0),
}


def price_for(model: str) -> Price:
    if model in PRICES:
        return PRICES[model]
    for k, p in PRICES.items():
        if model.startswith(k):
            return p
    # Unknown model: assume Sonnet-tier pricing rather than silently reporting zero.
    return PRICES["claude-sonnet-5-5"]


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float:
    p = price_for(model)
    return (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_read_input_tokens * p.cache_read
        + cache_creation_input_tokens * p.cache_write
    ) / 1_000_000
