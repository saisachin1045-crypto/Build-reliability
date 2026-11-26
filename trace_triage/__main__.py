"""Command line: python -m trace_triage demo | cost | sweep"""
from __future__ import annotations

import argparse
from typing import List, Optional

from . import costs
from .judge import TieredJudge, calibrate, cheapest_passing, self_consistency, simulated
from .pipeline import evaluate, format_evaluation, format_report, run
from .synth import generate


def _golden(seed: int, per_side: int = 150):
    pool = generate(6000, seed=seed + 1000)
    failing = [t for t in pool if t.truth.failure_mode][:per_side]
    healthy = [t for t in pool if not t.truth.failure_mode][:per_side]
    return failing + healthy


def _tiered(seed: int = 0) -> TieredJudge:
    return TieredJudge(simulated("mid", seed=seed), simulated("frontier", seed=seed))


def _calibration(seed: int) -> str:
    golden = _golden(seed)
    tiered = _tiered()
    rows = [("small", simulated("small"), "haiku-5.5"), ("mid", simulated("mid"), "sonnet-5.5"),
            ("frontier", simulated("frontier"), "opus-5.5"), ("mid->frontier", tiered, None)]
    lines = [
        f"Judge calibration on a golden set of {len(golden)} labelled traces "
        "(simulated judges: the error rates are assumptions, replace them with measured ones)",
        f"  {'judge':14} {'agreement':>9} {'recall':>7} {'precision':>9} {'class acc':>9}",
    ]
    for name, judge, _price in rows:
        c = calibrate(judge, golden)
        lines.append(f"  {name:14} {c.agreement:>9.0%} {c.failure_recall:>7.0%} {c.failure_precision:>9.0%} {c.class_accuracy:>9.0%}")
    escalation_rate = tiered.escalations / tiered.calls if tiered.calls else 0.0
    price = {t: costs.PRICES[t][0] for t in costs.PRICES}
    candidates = [
        (simulated("small"), price["haiku-5.5"]),
        (simulated("mid"), price["sonnet-5.5"]),
        (_tiered(), price["sonnet-5.5"] + escalation_rate * price["opus-5.5"]),
        (simulated("frontier"), price["opus-5.5"]),
    ]
    chosen = cheapest_passing(candidates, golden)
    flaky = self_consistency(lambda run: simulated("mid", run=run), golden)
    lines.append(f"  escalation rate of mid->frontier on the golden set: {escalation_rate:.0%}; mid judge flips its pass/fail verdict between two runs on {flaky:.0%} of traces")
    lines.append(
        "  cheapest judge clearing recall>=90%, precision>=90%, class accuracy>=80%: "
        + (chosen.tier if chosen else "none")
    )
    return "\n".join(lines)


def _demo(args) -> str:
    traces = generate(args.n, args.seed)
    judge = _tiered(args.seed) if args.judge == "tiered" else simulated(args.judge, seed=args.seed)
    report = run(traces, judge, seed=args.seed)
    return "\n\n".join(
        [
            _calibration(args.seed),
            format_report(report, judge),
            format_evaluation(evaluate(traces, report), report, traces),
        ]
    )


def _sweep(args) -> str:
    rows, surfaced_all, rooted_all = [], 0, 0
    per_mode = {}
    for seed in range(1, args.seeds + 1):
        traces = generate(args.n, seed)
        report = run(traces, _tiered(seed), seed=seed)
        results = evaluate(traces, report)
        s_all = all(r.surfaced for r in results)
        r_all = all(r.root_cause_ok for r in results)
        surfaced_all += s_all
        rooted_all += r_all
        for r in results:
            hit = per_mode.setdefault(r.mode, [0, 0])
            hit[0] += bool(r.root_cause_ok)
            hit[1] += 1
        rows.append(f"  seed {seed:>3}: all modes surfaced {'yes' if s_all else 'NO':>3}, every root cause right {'yes' if r_all else 'NO':>3}")
    lines = [f"{args.seeds} seeds, {args.n:,} traces each, mid->frontier judge"] + rows
    lines.append(f"all four modes surfaced in {surfaced_all}/{args.seeds} seeds; every top slice correct in {rooted_all}/{args.seeds} seeds")
    for mode, (hit, n) in per_mode.items():
        lines.append(f"  root cause right for {mode}: {hit}/{n}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="trace_triage", description="Cost-aware triage of agent traces.")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run the funnel on synthetic traces and score it against the seeded faults")
    demo.add_argument("--n", type=int, default=20000)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--judge", choices=["tiered", "small", "mid", "frontier"], default="tiered")
    sub.add_parser("cost", help="print the cost model for a million traces")
    sweep = sub.add_parser("sweep", help="repeat the demo over several seeds and report how often it gets every mode right")
    sweep.add_argument("--seeds", type=int, default=10)
    sweep.add_argument("--n", type=int, default=20000)
    args = parser.parse_args(argv)
    if args.command == "cost":
        print(costs.format_table())
    elif args.command == "demo":
        print(_demo(args))
    else:
        print(_sweep(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
