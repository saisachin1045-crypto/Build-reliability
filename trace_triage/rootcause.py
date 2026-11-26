"""Stage 4, the part that needs no model: slices, a change point and a failing-vs-passing contrast.

The analyses run on points: (trace, failed, weight). Two ways to get points:
  - judged samples, weighted back to the population (for silent failures, where only a judge knows);
  - exact labels on every trace with weight 1 (for loud failures, where a tripwire already labels
    the whole cluster for free, so nothing is sampled and the answer is not noisy).
They cost nothing. Only afterwards does a frontier model get involved, with a small, narrowed brief.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .models import Span, Trace
from .sampling import JudgedSample

Point = Tuple[Trace, bool, float]  # trace, did it fail in this way, how many traces it stands for


def _failed(sample: JudgedSample, failure_class: str) -> bool:
    return (not sample.verdict.goal_met) and sample.verdict.failure_class == failure_class


def sample_points(samples: Sequence[JudgedSample], failure_class: str) -> List[Point]:
    return [(s.trace, _failed(s, failure_class), s.weight) for s in samples]


@dataclass(frozen=True)
class Slice:
    dim: str
    value: str
    rate_in: float  # failure rate inside the slice
    rate_out: float  # failure rate everywhere else
    population: float  # estimated traces in the slice
    excess: float  # (rate_in - rate_out) * population: failures the slice explains
    n_sampled: int


def slice_points(points: Sequence[Point], dims: Sequence[str], min_n: int = 8, limit: int = 3) -> List[Slice]:
    """Attribute values where failures concentrate, ranked by failures explained."""
    total_w = sum(w for _, _, w in points)
    total_f = sum(w for _, failed, w in points if failed)
    found: List[Slice] = []
    if total_f <= 0:
        return found
    for dim in dims:
        groups = {}
        for point in points:
            value = point[0].attrs.get(dim)
            if value is not None:
                groups.setdefault(value, []).append(point)
        for value, members in groups.items():
            if len(members) < min_n:
                continue
            w_in = sum(w for _, _, w in members)
            f_in = sum(w for _, failed, w in members if failed)
            w_out, f_out = total_w - w_in, total_f - f_in
            if w_out <= 0:
                continue
            rate_in, rate_out = f_in / w_in, f_out / w_out
            if rate_in <= rate_out:
                continue
            found.append(Slice(dim, value, rate_in, rate_out, w_in, (rate_in - rate_out) * w_in, len(members)))
    # Equal failures explained: prefer the narrower slice (higher rate inside it). Rounding keeps float
    # noise from deciding the order. Slices can still be confounded when few failures were sampled;
    # a replay in stage 5 is what separates them.
    found.sort(key=lambda s: (round(s.excess, 6), s.rate_in), reverse=True)
    return found[:limit]


def top_slices(
    samples: Sequence[JudgedSample], failure_class: str, dims: Sequence[str], min_n: int = 8, limit: int = 3
) -> List[Slice]:
    return slice_points(sample_points(samples, failure_class), dims, min_n, limit)


@dataclass(frozen=True)
class Changepoint:
    ts: int
    rate_before: float
    rate_after: float


def _loglik(k: float, n: float) -> float:
    if n <= 0:
        return 0.0
    p = k / n
    return (k * math.log(p) if k > 0 else 0.0) + ((n - k) * math.log(1 - p) if n - k > 0 else 0.0)


def changepoint_points(
    points: Sequence[Point], min_side: float = 0.1, min_ratio: float = 2.0, min_score: float = 4.0
) -> Optional[Changepoint]:
    """The time after which failures got clearly more frequent, or None.

    Best single split by binomial likelihood. Weights are rescaled to the number of points so a
    large population does not make a weak signal look significant. Only increases are reported.
    """
    pts = sorted(points, key=lambda p: p[0].ts)
    if len(pts) < 10:
        return None
    scale = len(pts) / sum(w for _, _, w in pts)
    n_i = [w * scale for _, _, w in pts]
    k_i = [n if failed else 0.0 for (_, failed, _), n in zip(pts, n_i)]
    total_n, total_k = sum(n_i), sum(k_i)
    if total_k <= 0 or total_k >= total_n:
        return None
    base = _loglik(total_k, total_n)
    best = None
    n1 = k1 = 0.0
    for i in range(len(pts) - 1):
        n1 += n_i[i]
        k1 += k_i[i]
        if pts[i][0].ts == pts[i + 1][0].ts:
            continue
        n2, k2 = total_n - n1, total_k - k1
        if n1 < min_side * total_n or n2 < min_side * total_n:
            continue
        score = _loglik(k1, n1) + _loglik(k2, n2) - base
        if best is None or score > best[0]:
            best = (score, pts[i + 1][0].ts, k1 / n1, k2 / n2)
    if best is None:
        return None
    score, ts, before, after = best
    if score < min_score or after <= before or after < min_ratio * before:
        return None
    return Changepoint(ts, before, after)


def find_changepoint(samples: Sequence[JudgedSample], failure_class: str, **kwargs) -> Optional[Changepoint]:
    return changepoint_points(sample_points(samples, failure_class), **kwargs)


@dataclass(frozen=True)
class Contrast:
    failing_id: str
    passing_id: Optional[str]
    first_divergence: Optional[int]
    detail: str


def _signature(s: Span):
    return s.kind, s.name, s.status, s.args_valid, s.finish_reason


def _tools(trace: Trace):
    return frozenset((s.kind, s.name) for s in trace.spans if s.kind != "llm")


def _divergence(a: Trace, b: Trace) -> Optional[int]:
    for i, (x, y) in enumerate(zip(a.spans, b.spans)):
        if _signature(x) != _signature(y):
            return i
    if len(a.spans) != len(b.spans):
        return min(len(a.spans), len(b.spans))
    return None


def _describe(s: Span) -> str:
    text = f"{s.kind}:{s.name} {s.status}"
    if s.finish_reason != "stop":
        text += f" finish={s.finish_reason}"
    if not s.args_valid:
        text += " args=invalid"
    return text


def contrast_pair(failing: Trace, passing_pool: Sequence[Trace]) -> Contrast:
    """Align a failing trace with the passing trace that agrees with it longest; report where they split."""
    candidates = [t for t in passing_pool if _tools(t) == _tools(failing)]
    if not candidates:
        return Contrast(failing.trace_id, None, None, "no passing trace with the same tools to compare against")

    def agreement(t: Trace):
        d = _divergence(failing, t)
        prefix = d if d is not None else len(failing.spans) + 1  # identical structure agrees longest
        return prefix, -sum(1 for s in t.spans if s.status != "ok"), -abs(len(t.spans) - len(failing.spans))

    best = max(candidates, key=agreement)
    d = _divergence(failing, best)
    if d is None:
        detail = "no structural divergence: the spans match, so the failure is in content and needs a content-level review"
    else:
        f = _describe(failing.spans[d]) if d < len(failing.spans) else "<no span>"
        p = _describe(best.spans[d]) if d < len(best.spans) else "<no span>"
        detail = f"span {d}: failing {f} vs passing {p}"
    return Contrast(failing.trace_id, best.trace_id, d, detail)
