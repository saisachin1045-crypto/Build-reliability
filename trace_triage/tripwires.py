"""Stage 1: cheap tripwires computed from span metadata only. No model, no ground truth.

A tripwire says where to look, never why. It also misses silent failures, which is why
stage 3 audits the traces that tripped nothing.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import median
from typing import Dict, FrozenSet, Iterable

from .fingerprint import shape_key
from .models import Trace

# Flags strong enough to name a cluster when there is no error text.
LOUD_FLAGS = ("truncated", "invalid_args", "no_final_answer", "loop")
USER_FLAGS = {
    "thumbs_down": "user_thumbs_down",
    "rephrase": "user_rephrase",
    "escalated": "user_escalated",
    "abandoned": "user_abandoned",
}


def total_tokens(trace: Trace) -> int:
    return sum(s.tokens_in + s.tokens_out for s in trace.spans)


def total_latency(trace: Trace) -> int:
    return sum(s.latency_ms for s in trace.spans)


@dataclass(frozen=True)
class ShapeStats:
    """Median tokens and latency per trace shape: the baseline for outlier flags."""

    tokens: Dict[str, float]
    latency: Dict[str, float]

    @classmethod
    def fit(cls, traces: Iterable[Trace]) -> "ShapeStats":
        tokens: Dict[str, list] = {}
        latency: Dict[str, list] = {}
        for t in traces:
            key = shape_key(t)
            tokens.setdefault(key, []).append(total_tokens(t))
            latency.setdefault(key, []).append(total_latency(t))
        return cls(
            tokens={k: median(v) for k, v in tokens.items()},
            latency={k: median(v) for k, v in latency.items()},
        )


def compute_flags(trace: Trace, stats: ShapeStats, outlier_factor: float = 3.0) -> FrozenSet[str]:
    flags = set()
    spans = trace.spans
    if any(s.status != "ok" for s in spans):
        flags.add("error_span")
    if any(s.status == "timeout" for s in spans):
        flags.add("timeout")
    if any(not s.args_valid for s in spans):
        flags.add("invalid_args")
    if any(s.finish_reason == "length" for s in spans):
        flags.add("truncated")
    if trace.final_answer is None:
        flags.add("no_final_answer")

    repeats = max(Counter((s.name, s.args_hash) for s in spans if s.kind == "tool").values(), default=0)
    if repeats >= 2:
        flags.add("retry")
    if repeats >= 3:
        flags.add("loop")

    key = shape_key(trace)
    if stats.tokens.get(key) and total_tokens(trace) > outlier_factor * stats.tokens[key]:
        flags.add("token_outlier")
    if stats.latency.get(key) and total_latency(trace) > outlier_factor * stats.latency[key]:
        flags.add("latency_outlier")

    if trace.user_signal in USER_FLAGS:
        flags.add(USER_FLAGS[trace.user_signal])
    return frozenset(flags)
