"""Stage 3 judge: the interface, a simulated stand-in, tier escalation and golden-set calibration.

This repo never calls a model. SimulatedJudge reads ground truth and adds configurable mistakes,
so the rest of the pipeline can be tested against judges of different quality. The numbers in
TIERS are illustrative assumptions, not measurements: measure your own judges with calibrate().
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from .models import Trace
from .stats import wilson_interval
from .synth import FAILURE_MODES


@dataclass(frozen=True)
class Verdict:
    goal_met: bool
    failure_class: Optional[str] = None
    first_bad_span: Optional[int] = None
    confidence: float = 1.0
    tier: str = ""


class Judge(Protocol):
    tier: str

    def judge(self, trace: Trace) -> Verdict: ...


@dataclass
class SimulatedJudge:
    """Stand-in for an LLM judge. Verdicts depend on (tier, seed, run, trace id), never on call order.

    The tier is part of the random stream so two tiers make independent mistakes on the same trace,
    which is what makes escalating a low-confidence verdict to a stronger tier worth anything.
    """

    tier: str
    recall: float
    false_positive_rate: float
    class_accuracy: float
    seed: int = 0
    run: int = 0

    def judge(self, trace: Trace) -> Verdict:
        rng = random.Random(f"{self.tier}:{self.seed}:{self.run}:{trace.trace_id}")
        truth = trace.truth
        mode = truth.failure_mode if truth else None
        if mode:
            if rng.random() >= self.recall:
                return Verdict(True, None, None, 0.6, self.tier)  # a missed failure
            if rng.random() < self.class_accuracy:
                return Verdict(False, mode, truth.first_bad_span, 0.9, self.tier)
            wrong = rng.choice([m for m in FAILURE_MODES if m != mode] + ["other"])
            return Verdict(False, wrong, None, 0.55, self.tier)
        if rng.random() < self.false_positive_rate:
            return Verdict(False, rng.choice(list(FAILURE_MODES) + ["other"]), None, 0.5, self.tier)
        return Verdict(True, None, None, 0.9, self.tier)


# Illustrative only. Replace with numbers measured on your own golden set.
TIERS: Dict[str, Dict[str, float]] = {
    "small": dict(recall=0.55, false_positive_rate=0.05, class_accuracy=0.60),
    "mid": dict(recall=0.88, false_positive_rate=0.03, class_accuracy=0.85),
    "frontier": dict(recall=0.96, false_positive_rate=0.01, class_accuracy=0.96),
}


def simulated(tier: str, seed: int = 0, run: int = 0) -> SimulatedJudge:
    return SimulatedJudge(tier, seed=seed, run=run, **TIERS[tier])


class TieredJudge:
    """Ask the primary judge; send low-confidence verdicts one tier up."""

    def __init__(self, primary: Judge, escalation: Judge, threshold: float = 0.7):
        self.primary, self.escalation, self.threshold = primary, escalation, threshold
        self.escalations = 0
        self.calls = 0

    @property
    def tier(self) -> str:
        return f"{self.primary.tier}->{self.escalation.tier}"

    def judge(self, trace: Trace) -> Verdict:
        self.calls += 1
        verdict = self.primary.judge(trace)
        if verdict.confidence < self.threshold:
            self.escalations += 1
            return self.escalation.judge(trace)
        return verdict


@dataclass(frozen=True)
class Calibration:
    n: int
    agreement: float
    failure_recall: float
    failure_precision: float
    class_accuracy: float
    per_class_recall: Dict[str, float]
    # 95% lower bounds: what the golden set can actually prove, which shrinks when it is small
    failure_recall_lo: float = 0.0
    failure_precision_lo: float = 0.0
    class_accuracy_lo: float = 0.0


def calibrate(judge: Judge, golden: Sequence[Trace]) -> Calibration:
    """Score a judge against hand-labelled traces (here: traces carrying ground truth)."""
    tp = fp = fn = tn = class_ok = 0
    per_class: Dict[str, List[int]] = {}
    for t in golden:
        mode = t.truth.failure_mode
        verdict = judge.judge(t)
        flagged = not verdict.goal_met
        if mode:
            bucket = per_class.setdefault(mode, [0, 0])
            bucket[1] += 1
            if flagged:
                tp += 1
                if verdict.failure_class == mode:
                    class_ok += 1
                    bucket[0] += 1
            else:
                fn += 1
        elif flagged:
            fp += 1
        else:
            tn += 1
    n = len(golden)
    if tp:
        class_accuracy = class_ok / tp
    else:
        class_accuracy = 0.0 if fn else 1.0
    return Calibration(
        n=n,
        agreement=(tp + tn) / n if n else 1.0,
        failure_recall=tp / (tp + fn) if tp + fn else 1.0,
        failure_precision=tp / (tp + fp) if tp + fp else 1.0,
        class_accuracy=class_accuracy,
        per_class_recall={m: ok / total for m, (ok, total) in per_class.items()},
        failure_recall_lo=wilson_interval(tp, tp + fn)[0],
        failure_precision_lo=wilson_interval(tp, tp + fp)[0],
        class_accuracy_lo=wilson_interval(class_ok, tp)[0],
    )


def cheapest_passing(
    candidates: Sequence[Tuple[Judge, float]],
    golden: Sequence[Trace],
    *,
    min_recall: float = 0.9,
    min_precision: float = 0.9,
    min_class_accuracy: float = 0.8,
) -> Optional[Judge]:
    """The cheapest judge that clears the bar on the golden set, or None.

    A judge clears the bar only if the 95% lower bound does, not the measured value: a small golden
    set cannot certify a judge whose measured score merely sits near the bar.
    """
    for judge, _cost in sorted(candidates, key=lambda c: c[1]):
        c = calibrate(judge, golden)
        if (
            c.failure_recall_lo >= min_recall
            and c.failure_precision_lo >= min_precision
            and c.class_accuracy_lo >= min_class_accuracy
        ):
            return judge
    return None


def self_consistency(make_judge: Callable[[int], Judge], traces: Sequence[Trace], runs: int = 2) -> float:
    """Share of traces whose pass/fail verdict changes between runs: the judge's own flakiness."""
    judges = [make_judge(r) for r in range(runs)]
    changed = sum(1 for t in traces if len({j.judge(t).goal_met for j in judges}) > 1)
    return changed / len(traces) if traces else 0.0
