"""Cost model for the funnel, and for the alternative of judging every trace.

Every number is an assumption you can change in Assumptions. Prices are vendor list prices as
cached on 2026-11-03 (USD per 1M tokens, input and output); check them before relying on them.
Batch processing is 50% off every token and stacks with caching. Thinking tokens bill as output,
so the output-token assumption for stage 3 is a lower bound if the judge runs with thinking on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

PRICES = {
    "haiku-5.5": (0.10, 0.50),
    "sonnet-5.5": (2.00, 10.00),
    "opus-5.5": (4.00, 20.00),
    "fable-5.2": (8.00, 40.00),
}
BATCH_DISCOUNT = 0.5


@dataclass(frozen=True)
class Assumptions:
    n_traces: int = 1_000_000
    raw_tokens: int = 60_000  # per trace: agents resend their history on every call
    compressed_tokens: int = 4_000  # per trace after dedupe and compression, what a judge reads
    clusters: int = 2_000
    per_cluster: int = 10
    clean_audit: int = 2_000
    out_tokens_per_verdict: int = 300
    embed_tokens_per_trace: int = 500  # first user message plus final answer
    embed_price_per_m: Tuple[float, float] = (0.02, 0.13)
    tripwire_compute_usd: float = 20.0
    top_n: int = 40
    rca_per_incident: Tuple[float, float] = (2.5, 7.5)  # hard-capped agentic investigation
    verify_per_incident: Tuple[float, float] = (2.5, 5.0)  # replay before and after a fix

    @property
    def judged(self) -> int:
        return self.clusters * self.per_cluster + self.clean_audit


@dataclass(frozen=True)
class StageCost:
    name: str
    low: float
    high: float


def naive_cost(a: Assumptions, tier: str, batch: bool = False) -> float:
    """Reading every raw trace once on one tier (input tokens only: a lower bound)."""
    price_in, _ = PRICES[tier]
    cost = a.n_traces * a.raw_tokens * price_in / 1e6
    return cost * (BATCH_DISCOUNT if batch else 1.0)


def stage3_cost(a: Assumptions, tier: str, batch: bool = True) -> float:
    price_in, price_out = PRICES[tier]
    cost = (a.judged * a.compressed_tokens * price_in + a.judged * a.out_tokens_per_verdict * price_out) / 1e6
    return cost * (BATCH_DISCOUNT if batch else 1.0)


def estimate(a: Assumptions = Assumptions(), judge_tiers: Tuple[str, str] = ("sonnet-5.5", "opus-5.5")) -> List[StageCost]:
    low_tier, high_tier = judge_tiers
    embed_tokens = a.n_traces * a.embed_tokens_per_trace
    return [
        StageCost("1 tripwires (compute)", a.tripwire_compute_usd, a.tripwire_compute_usd),
        StageCost(
            "2 embeddings for clustering",
            embed_tokens * a.embed_price_per_m[0] / 1e6,
            embed_tokens * a.embed_price_per_m[1] / 1e6,
        ),
        StageCost("3 sampled judging (batch)", stage3_cost(a, low_tier), stage3_cost(a, high_tier)),
        StageCost("4 root cause, top incidents", a.top_n * a.rca_per_incident[0], a.top_n * a.rca_per_incident[1]),
        StageCost("5 verify fixes by replay", a.top_n * a.verify_per_incident[0], a.top_n * a.verify_per_incident[1]),
    ]


def total(stages: Sequence[StageCost]) -> StageCost:
    return StageCost("TOTAL", sum(s.low for s in stages), sum(s.high for s in stages))


def format_table(a: Assumptions = Assumptions()) -> str:
    stages = estimate(a)
    t = total(stages)
    lines = [
        f"Cost model for {a.n_traces:,} traces ({a.raw_tokens:,} raw tokens each, {a.judged:,} judged). "
        "Assumptions, not measurements.",
    ]
    for s in stages + [t]:
        lines.append(f"  {s.name:30} ${s.low:>9,.0f} - ${s.high:>9,.0f}")
    lines.append(f"  at most ${t.high / a.n_traces:.5f} per trace")
    lines.append("")
    lines.append("For comparison, judging every raw trace instead (input tokens only):")
    for tier in ("haiku-5.5", "sonnet-5.5", "opus-5.5"):
        lines.append(
            f"  {tier:12} ${naive_cost(a, tier):>10,.0f}   on batch ${naive_cost(a, tier, batch=True):>10,.0f}"
        )
    return "\n".join(lines)
