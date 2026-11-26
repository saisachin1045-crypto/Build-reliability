"""The funnel end to end, plus an evaluation that scores it against the seeded faults.

  stage 0  join traces into sessions
  stage 1  tripwires from span metadata (no model)
  stage 2  cluster by shape and error signature
  stage 3  sample each cluster, judge the sample, estimate the cluster; audit the clean pool
  stage 4  rank clusters, then slices + change point + contrast per incident (no model);
           Incident.brief() is the small, already-narrowed input a frontier model would get
"""
from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence

from .clustering import cluster, cluster_key
from .compress import compress
from .fingerprint import workflow_key
from .judge import Judge
from .models import Trace
from .rootcause import Changepoint, Contrast, Slice, changepoint_points, contrast_pair, sample_points, slice_points
from .sampling import AuditResult, ClusterEstimate, JudgedSample, audit_clean_pool, estimate_cluster
from .sessions import join_sessions
from .synth import DRIVERS, FAILURE_MODES
from .tripwires import ShapeStats, compute_flags

DIMS = ("prompt_version", "model", "deploy", "tier", "locale", "input_len")


@dataclass(frozen=True)
class Incident:
    rank: int
    estimate: ClusterEstimate
    failure_class: str
    slices: List[Slice]
    changepoint: Optional[Changepoint]
    contrast: Contrast
    example: str  # a compressed failing trace
    evidence: str  # where slices and the changepoint came from: exact labels or judged samples

    def brief(self) -> dict:
        e = self.estimate
        return {
            "cluster": {"shape": e.key[0], "signature": e.key[1], "size": e.size},
            "failure_class": self.failure_class,
            "evidence": self.evidence,
            "estimated_failures": round(e.est_failures),
            "failure_rate_lo_mid_hi": [round(e.lo, 3), round(e.rate, 3), round(e.hi, 3)],
            "slices": [f"{s.dim}={s.value} ({s.rate_in:.0%} vs {s.rate_out:.0%} elsewhere)" for s in self.slices],
            "changepoint": None
            if self.changepoint is None
            else {"ts": self.changepoint.ts, "before": round(self.changepoint.rate_before, 3), "after": round(self.changepoint.rate_after, 3)},
            "contrast": self.contrast.detail,
            "example": self.example,
        }


@dataclass(frozen=True)
class Report:
    n_traces: int
    n_flagged: int
    judged: int
    clusters: List[ClusterEstimate]
    audit: AuditResult
    incidents: List[Incident]
    flags: Dict[str, FrozenSet[str]]


def run(
    traces: Sequence[Trace],
    judge: Judge,
    *,
    top_n: int = 8,
    base: int = 10,
    cap: int = 60,
    ub_target: float = 0.06,
    audit_n: int = 1000,
    min_rate: float = 0.02,
    min_failures: int = 20,
    min_pass_confidence: float = 0.8,
    exact_label_rate: float = 0.8,
    seed: int = 0,
    dims: Sequence[str] = DIMS,
) -> Report:
    rng = random.Random(seed)
    traces = join_sessions(traces)
    stats = ShapeStats.fit(traces)
    flags = {t.trace_id: compute_flags(t, stats) for t in traces}
    groups = cluster(traces, flags)

    estimates: List[ClusterEstimate] = []
    by_workflow: Dict[str, List[JudgedSample]] = defaultdict(list)
    traces_by_workflow: Dict[str, List[Trace]] = defaultdict(list)
    key_of = {}
    workflow_of = {}
    all_samples: List[JudgedSample] = []
    judged = 0
    for key, members in groups.items():
        estimate, samples = estimate_cluster(key, members, flags, judge, rng, base, cap, ub_target)
        estimates.append(estimate)
        workflow_of[key] = workflow_key(members[0])
        by_workflow[workflow_of[key]].extend(samples)
        traces_by_workflow[workflow_of[key]].extend(members)
        key_of.update({t.trace_id: key for t in members})
        all_samples.extend(samples)
        judged += estimate.judged

    audit = audit_clean_pool([t for t in traces if not flags[t.trace_id]], judge, rng, audit_n)
    judged += audit.judged

    candidates = [e for e in estimates if e.rate >= min_rate and e.est_failures >= min_failures and e.classes]
    candidates.sort(key=lambda e: e.est_failures, reverse=True)
    # Contrast only against passes the judge was confident about: a missed failure would otherwise
    # sit in the pool looking like a healthy example and hide the real divergence.
    passing = [s.trace for s in all_samples if s.verdict.goal_met and s.verdict.confidence >= min_pass_confidence]
    incidents: List[Incident] = []
    for rank, est in enumerate(candidates[:top_n], start=1):
        failure_class = max(est.classes, key=est.classes.get)
        workflow = workflow_of[est.key]
        workflow_samples = by_workflow[workflow]
        failing_here = [
            s.trace
            for s in workflow_samples
            if not s.verdict.goal_met and s.verdict.failure_class == failure_class and key_of[s.trace.trace_id] == est.key
        ]
        example = failing_here[0]
        # A loud cluster is defined by a tripwire, so every trace in the workflow is already labelled
        # (in the cluster or not) for free: use all of them, nothing is sampled. A silent cluster has
        # no such label, so fall back to the judged samples, weighted back to the population.
        if est.key[1] != "-" and est.rate >= exact_label_rate:
            points = [(t, key_of[t.trace_id] == est.key, 1.0) for t in traces_by_workflow[workflow]]
            evidence = f"exact labels on {len(points):,} traces"
        else:
            points = sample_points(workflow_samples, failure_class)
            evidence = f"{len(points)} judged samples, weighted"
        incidents.append(
            Incident(
                rank=rank,
                estimate=est,
                failure_class=failure_class,
                slices=slice_points(points, dims),
                changepoint=changepoint_points(points),
                contrast=contrast_pair(example, passing),
                example=compress(example),
                evidence=evidence,
            )
        )
    return Report(
        n_traces=len(traces),
        n_flagged=sum(1 for f in flags.values() if f),
        judged=judged,
        clusters=estimates,
        audit=audit,
        incidents=incidents,
        flags=flags,
    )


@dataclass(frozen=True)
class ModeResult:
    mode: str
    failing: int  # true failing traces of this mode
    any_flag_share: float  # share that tripped at least one wire, user signals included
    metadata_share: float  # share that tripped a wire not based on user signals
    surfaced: bool  # an incident has this mode as its dominant class
    est_failures: Optional[float]
    root_cause_ok: Optional[bool]  # the top slice is the seeded driver


def evaluate(traces: Sequence[Trace], report: Report) -> List[ModeResult]:
    """Score the run against ground truth. The only place besides the simulated judge that reads it."""
    results = []
    for mode in FAILURE_MODES:
        failing = [t for t in traces if t.truth and t.truth.failure_mode == mode]
        n = len(failing)
        any_share = sum(1 for t in failing if report.flags[t.trace_id]) / n if n else 0.0
        meta_share = (
            sum(1 for t in failing if any(not f.startswith("user_") for f in report.flags[t.trace_id])) / n
            if n
            else 0.0
        )
        matches = [i for i in report.incidents if i.failure_class == mode]
        best = max(matches, key=lambda i: i.estimate.est_failures, default=None)
        ok = None
        if best is not None:
            ok = bool(best.slices) and (best.slices[0].dim, best.slices[0].value) == DRIVERS[mode]
        results.append(
            ModeResult(mode, n, any_share, meta_share, best is not None, best.estimate.est_failures if best else None, ok)
        )
    return results


def missed_by_tripwires(traces: Sequence[Trace], report: Report) -> int:
    """True failures that tripped no wire at all (what the clean-pool audit tries to size)."""
    return sum(1 for t in traces if t.truth and t.truth.failure_mode and not report.flags[t.trace_id])


def format_report(report: Report, judge=None) -> str:
    n = report.n_traces
    share = (lambda x: x / n) if n else (lambda x: 0.0)
    lines = [
        f"Traces {n:,} | tripped a wire {report.n_flagged:,} ({share(report.n_flagged):.1%}) | "
        f"clusters {len(report.clusters)} | judged {report.judged:,} ({share(report.judged):.1%})"
    ]
    if judge is not None and hasattr(judge, "escalations"):
        lines.append(f"Judge {judge.tier}: {judge.calls:,} calls, {judge.escalations:,} escalated one tier up")
    a = report.audit
    lines.append(
        f"Clean-pool audit: {a.judged:,} of {a.population:,} untripped traces judged, {a.failures} failed -> "
        f"{a.rate:.1%} [{a.lo:.1%}, {a.hi:.1%}], about {a.est_missed:,.0f} silent failures"
    )
    lines += ["", "Incidents"]
    for i in report.incidents:
        e = i.estimate
        lines.append(f"#{i.rank} {i.failure_class}   cluster: {e.key[0]} | {e.key[1]}")
        lines.append(
            f"    size {e.size:,}, failure rate {e.rate:.1%} [{e.lo:.1%}, {e.hi:.1%}], "
            f"about {e.est_failures:,.0f} failures ({e.judged} judged)"
        )
        if i.slices:
            s = i.slices[0]
            lines.append(
                f"    top slice: {s.dim}={s.value} ({s.rate_in:.0%} vs {s.rate_out:.0%} elsewhere, explains ~{s.excess:,.0f}; {i.evidence})"
            )
        if i.changepoint:
            c = i.changepoint
            lines.append(f"    changepoint: ts {c.ts:,} ({c.rate_before:.0%} -> {c.rate_after:.0%})")
        lines.append(f"    contrast: {i.contrast.detail}")
    return "\n".join(lines)


def format_evaluation(results: Sequence[ModeResult], report: Report, traces: Sequence[Trace]) -> str:
    def yes(flag):
        return "-" if flag is None else ("yes" if flag else "NO")

    lines = [
        "Scored against the seeded failure modes (ground truth, never read by stages 1, 2 and 4)",
        f"  {'mode':24} {'failing':>7} {'any wire':>9} {'metadata':>9} {'surfaced':>9} {'est':>7} {'root cause':>11}",
    ]
    for r in results:
        est = "-" if r.est_failures is None else f"{r.est_failures:,.0f}"
        lines.append(
            f"  {r.mode:24} {r.failing:>7,} {r.any_flag_share:>9.0%} {r.metadata_share:>9.0%} "
            f"{yes(r.surfaced):>9} {est:>7} {yes(r.root_cause_ok):>11}"
        )
    missed = missed_by_tripwires(traces, report)
    lines.append(f"  failures that tripped no wire: {missed:,} (the audit estimated {report.audit.est_missed:,.0f})")
    return "\n".join(lines)
