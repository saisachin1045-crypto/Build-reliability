"""Stage 3 sampling: judge a few traces per cluster, then estimate the whole cluster honestly.

Each cluster is split into a flagged stratum and a clean stratum. Flagged traces are over-sampled
on purpose, so every sample carries a weight (stratum size / samples drawn) and estimates are
weighted back to the population. A stratum keeps widening until it is clearly mostly failing,
clearly mostly fine (upper bound under ub_target), or hits the cap.
"""
from __future__ import annotations

import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import FrozenSet, List, Mapping, Sequence, Tuple

from .clustering import ClusterKey
from .judge import Judge, Verdict
from .models import Trace
from .stats import wilson_interval  # noqa: F401  (also re-exported for callers of this module)


@dataclass(frozen=True)
class Stratum:
    size: int
    sampled: int
    failures: int


def stratified_estimate(strata: Sequence[Stratum], z: float = 1.96) -> Tuple[float, float, float]:
    """Population failure rate and a 95% interval from strata sampled at different rates."""
    strata = [s for s in strata if s.size > 0]
    total = sum(s.size for s in strata)
    if total == 0:
        return 0.0, 0.0, 0.0
    rate = variance = 0.0
    for s in strata:
        if s.sampled == 0:
            raise ValueError("every non-empty stratum needs at least one sample")
        w = s.size / total
        rate += w * s.failures / s.sampled
        smoothed = (s.failures + 0.5) / (s.sampled + 1)  # keeps 0/n and n/n from claiming certainty
        finite_population = 1 - s.sampled / s.size
        variance += w * w * smoothed * (1 - smoothed) / s.sampled * finite_population
    se = math.sqrt(variance)
    return rate, max(0.0, rate - z * se), min(1.0, rate + z * se)


@dataclass(frozen=True)
class JudgedSample:
    trace: Trace
    verdict: Verdict
    weight: float  # how many traces in the population this sample stands for


@dataclass(frozen=True)
class ClusterEstimate:
    key: ClusterKey
    size: int
    judged: int
    rate: float
    lo: float
    hi: float
    est_failures: float
    classes: Mapping[str, int]  # failure class -> judged failures


def _judge_stratum(population, judge, rng, base, cap, ub_target):
    order = rng.sample(population, len(population))  # one random order; widening extends the same sample
    drawn: List[Tuple[Trace, Verdict]] = []
    take = min(base, len(order))
    while True:
        drawn.extend((t, judge.judge(t)) for t in order[len(drawn) : len(drawn) + take])
        n = len(drawn)
        k = sum(1 for _, v in drawn if not v.goal_met)
        lo, hi = wilson_interval(k, n)
        if n >= len(order) or n >= cap or lo >= 0.5 or hi <= ub_target:
            return drawn
        take = min(base, len(order) - n, cap - n)


def estimate_cluster(
    key: ClusterKey,
    members: Sequence[Trace],
    flags: Mapping[str, FrozenSet[str]],
    judge: Judge,
    rng: random.Random,
    base: int = 10,
    cap: int = 60,
    ub_target: float = 0.06,
) -> Tuple[ClusterEstimate, List[JudgedSample]]:
    flagged = [t for t in members if flags[t.trace_id]]
    clean = [t for t in members if not flags[t.trace_id]]
    populations = [p for p in (flagged, clean) if p]
    per_stratum = max(1, base // max(1, len(populations)))
    strata: List[Stratum] = []
    samples: List[JudgedSample] = []
    for population in populations:
        drawn = _judge_stratum(population, judge, rng, per_stratum, cap, ub_target)
        failures = sum(1 for _, v in drawn if not v.goal_met)
        strata.append(Stratum(len(population), len(drawn), failures))
        weight = len(population) / len(drawn)
        samples.extend(JudgedSample(t, v, weight) for t, v in drawn)
    rate, lo, hi = stratified_estimate(strata)
    classes = Counter(s.verdict.failure_class or "unclassified" for s in samples if not s.verdict.goal_met)
    estimate = ClusterEstimate(key, len(members), len(samples), rate, lo, hi, rate * len(members), classes)
    return estimate, samples


@dataclass(frozen=True)
class AuditResult:
    population: int
    judged: int
    failures: int
    rate: float
    lo: float
    hi: float
    est_missed: float  # failures hiding in traces that tripped nothing


def audit_clean_pool(pool: Sequence[Trace], judge: Judge, rng: random.Random, n: int) -> AuditResult:
    """Judge a random sample of traces that tripped no wire: the only way to measure what tripwires miss."""
    if not pool:
        return AuditResult(0, 0, 0, 0.0, 0.0, 0.0, 0.0)
    chosen = rng.sample(list(pool), min(n, len(pool)))
    failures = sum(1 for t in chosen if not judge.judge(t).goal_met)
    rate = failures / len(chosen)
    lo, hi = wilson_interval(failures, len(chosen))
    return AuditResult(len(pool), len(chosen), failures, rate, lo, hi, rate * len(pool))
