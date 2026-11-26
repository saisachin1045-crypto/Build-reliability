import unittest

from tests.helpers import failing_truth, make_trace
from trace_triage.judge import (
    SimulatedJudge,
    TieredJudge,
    Verdict,
    calibrate,
    cheapest_passing,
    self_consistency,
)
from trace_triage.models import Truth
from trace_triage.synth import FAILURE_MODES


def golden(n_fail=60, n_ok=60):
    failing = [
        make_trace(f"f{i}", truth=failing_truth(FAILURE_MODES[i % len(FAILURE_MODES)], first_bad_span=1))
        for i in range(n_fail)
    ]
    healthy = [make_trace(f"h{i}", truth=Truth(None)) for i in range(n_ok)]
    return failing + healthy


def perfect():
    return SimulatedJudge("perfect", recall=1.0, false_positive_rate=0.0, class_accuracy=1.0)


class SimulatedJudgeTests(unittest.TestCase):
    def test_perfect_judge_reproduces_the_truth_for_a_failing_trace(self):
        trace = make_trace(truth=failing_truth("bad_tool_args", first_bad_span=3))
        verdict = perfect().judge(trace)
        self.assertFalse(verdict.goal_met)
        self.assertEqual((verdict.failure_class, verdict.first_bad_span), ("bad_tool_args", 3))

    def test_perfect_judge_passes_a_healthy_trace(self):
        self.assertTrue(perfect().judge(make_trace(truth=Truth(None))).goal_met)

    def test_zero_recall_judge_misses_every_failure(self):
        judge = SimulatedJudge("blind", recall=0.0, false_positive_rate=0.0, class_accuracy=1.0)
        self.assertTrue(all(judge.judge(t).goal_met for t in golden()))

    def test_false_positive_rate_applies_to_healthy_traces(self):
        judge = SimulatedJudge("jumpy", recall=1.0, false_positive_rate=1.0, class_accuracy=1.0)
        self.assertFalse(judge.judge(make_trace(truth=Truth(None))).goal_met)

    def test_a_wrong_class_is_never_the_true_class(self):
        judge = SimulatedJudge("sloppy", recall=1.0, false_positive_rate=0.0, class_accuracy=0.0)
        for t in golden(n_fail=40, n_ok=0):
            self.assertNotEqual(judge.judge(t).failure_class, t.truth.failure_mode)

    def test_verdict_depends_on_trace_and_run_not_on_call_order(self):
        judge = SimulatedJudge("noisy", recall=0.5, false_positive_rate=0.1, class_accuracy=0.7)
        traces = golden()
        first = [judge.judge(t) for t in traces]
        again = [judge.judge(t) for t in reversed(traces)][::-1]
        self.assertEqual(first, again)

    def test_different_tiers_make_independent_mistakes_on_the_same_trace(self):
        a = SimulatedJudge("a", 0.5, 0.2, 0.7)
        b = SimulatedJudge("b", 0.5, 0.2, 0.7)  # same settings, different tier name
        traces = golden()
        pattern = lambda judge: [(v.goal_met, v.failure_class) for v in map(judge.judge, traces)]
        self.assertNotEqual(pattern(a), pattern(b))

    def test_a_different_run_can_change_the_verdict(self):
        run0 = SimulatedJudge("noisy", 0.5, 0.1, 0.7, run=0)
        run1 = SimulatedJudge("noisy", 0.5, 0.1, 0.7, run=1)
        traces = golden()
        self.assertNotEqual([run0.judge(t) for t in traces], [run1.judge(t) for t in traces])


class StubJudge:
    def __init__(self, goal_met, confidence, tier):
        self._v = Verdict(goal_met, None if goal_met else "x", None, confidence, tier)
        self.tier = tier

    def judge(self, trace):
        return self._v


class TieredJudgeTests(unittest.TestCase):
    def test_a_confident_primary_verdict_is_kept(self):
        tiered = TieredJudge(StubJudge(True, 0.9, "mid"), StubJudge(False, 1.0, "frontier"), threshold=0.7)
        verdict = tiered.judge(make_trace())
        self.assertEqual(verdict.tier, "mid")
        self.assertEqual(tiered.escalations, 0)

    def test_a_low_confidence_verdict_goes_to_the_escalation_judge(self):
        tiered = TieredJudge(StubJudge(True, 0.4, "mid"), StubJudge(False, 1.0, "frontier"), threshold=0.7)
        verdict = tiered.judge(make_trace())
        self.assertEqual((verdict.tier, verdict.goal_met), ("frontier", False))
        self.assertEqual((tiered.escalations, tiered.calls), (1, 1))


class CalibrationTests(unittest.TestCase):
    def test_a_perfect_judge_scores_one_everywhere(self):
        c = calibrate(perfect(), golden())
        self.assertEqual(
            (c.agreement, c.failure_recall, c.failure_precision, c.class_accuracy), (1.0, 1.0, 1.0, 1.0)
        )
        self.assertTrue(all(v == 1.0 for v in c.per_class_recall.values()))

    def test_a_blind_judge_has_zero_recall_but_still_agrees_on_healthy_traces(self):
        blind = SimulatedJudge("blind", 0.0, 0.0, 1.0)
        c = calibrate(blind, golden())
        self.assertEqual(c.failure_recall, 0.0)
        self.assertEqual(c.agreement, 0.5)

    def test_class_accuracy_counts_only_detected_failures(self):
        sloppy = SimulatedJudge("sloppy", recall=1.0, false_positive_rate=0.0, class_accuracy=0.0)
        c = calibrate(sloppy, golden())
        self.assertEqual((c.failure_recall, c.class_accuracy), (1.0, 0.0))

    def test_a_golden_set_without_failures_does_not_divide_by_zero(self):
        c = calibrate(perfect(), golden(n_fail=0, n_ok=10))
        self.assertEqual((c.failure_recall, c.failure_precision), (1.0, 1.0))

    def test_cheapest_passing_picks_the_cheapest_judge_that_clears_the_bar(self):
        weak = SimulatedJudge("weak", 0.4, 0.05, 0.5)
        strong = perfect()
        chosen = cheapest_passing([(strong, 5.0), (weak, 1.0)], golden())
        self.assertIs(chosen, strong)
        also_ok = SimulatedJudge("cheap-but-good", 1.0, 0.0, 1.0)
        self.assertIs(cheapest_passing([(strong, 5.0), (also_ok, 1.0)], golden()), also_ok)

    def test_calibration_reports_lower_bounds_that_shrink_with_a_small_golden_set(self):
        small = calibrate(perfect(), golden(n_fail=5, n_ok=5))
        large = calibrate(perfect(), golden(n_fail=100, n_ok=100))
        self.assertEqual(small.failure_recall, 1.0)
        self.assertLess(small.failure_recall_lo, 0.6)
        self.assertGreater(large.failure_recall_lo, 0.95)
        self.assertGreater(large.failure_precision_lo, 0.95)
        self.assertGreater(large.class_accuracy_lo, 0.95)

    def test_a_perfect_judge_on_a_tiny_golden_set_cannot_prove_it_clears_the_bar(self):
        self.assertIsNone(cheapest_passing([(perfect(), 1.0)], golden(n_fail=5, n_ok=5)))

    def test_a_judge_that_only_barely_clears_the_bar_on_the_point_estimate_does_not_pass(self):
        class DetectsFirst:
            tier = "scripted"

            def __init__(self, n):
                self.n = n

            def judge(self, trace):
                index = int(trace.trace_id[1:])
                if trace.truth.failure_mode and index < self.n:
                    return Verdict(False, trace.truth.failure_mode, None, 1.0, "scripted")
                return Verdict(True, None, None, 1.0, "scripted")

        # 91 of 100 failures found: point recall 0.91 clears a 0.90 bar, the lower bound (about 0.84) does not
        self.assertIsNone(cheapest_passing([(DetectsFirst(91), 1.0)], golden(n_fail=100, n_ok=100)))
        self.assertIsNotNone(cheapest_passing([(DetectsFirst(100), 1.0)], golden(n_fail=100, n_ok=100)))

    def test_cheapest_passing_returns_none_when_nobody_clears_the_bar(self):
        weak = SimulatedJudge("weak", 0.4, 0.05, 0.5)
        self.assertIsNone(cheapest_passing([(weak, 1.0)], golden()))


class SelfConsistencyTests(unittest.TestCase):
    def test_a_deterministic_judge_has_zero_flakiness(self):
        self.assertEqual(self_consistency(lambda run: perfect(), golden()), 0.0)

    def test_a_noisy_judge_changes_its_mind_between_runs(self):
        flaky = self_consistency(lambda run: SimulatedJudge("noisy", 0.5, 0.2, 0.7, run=run), golden())
        self.assertGreater(flaky, 0.05)


if __name__ == "__main__":
    unittest.main()
