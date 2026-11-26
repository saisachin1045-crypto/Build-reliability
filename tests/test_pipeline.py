import json
import unittest

from trace_triage.compress import estimate_tokens
from trace_triage.judge import TieredJudge, simulated
from trace_triage.pipeline import evaluate, format_evaluation, format_report, missed_by_tripwires, run
from trace_triage.synth import FAILURE_MODES, generate


def make_judge():
    return TieredJudge(simulated("mid"), simulated("frontier"))


def incident_for(report, mode):
    return next(i for i in report.incidents if i.failure_class == mode)


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = generate(20000, seed=7)
        cls.report = run(cls.traces, make_judge(), seed=1)
        cls.results = {r.mode: r for r in evaluate(cls.traces, cls.report)}

    def test_the_funnel_judges_a_small_fraction_of_the_traces(self):
        self.assertLess(self.report.judged, 0.15 * len(self.traces))

    def test_every_seeded_failure_mode_becomes_an_incident_of_the_right_class(self):
        for mode in FAILURE_MODES:
            self.assertTrue(self.results[mode].surfaced, mode)

    def test_the_top_slice_names_the_seeded_driver_for_every_mode(self):
        for mode in FAILURE_MODES:
            self.assertTrue(self.results[mode].root_cause_ok, mode)

    def test_loud_modes_trip_metadata_wires_but_the_silent_mode_does_not(self):
        for mode in ("tool_timeout_loop", "bad_tool_args", "context_truncated"):
            self.assertGreater(self.results[mode].metadata_share, 0.99, mode)
        silent = self.results["wrong_customer_silent"]
        self.assertLess(silent.metadata_share, 0.05)
        self.assertLess(silent.any_flag_share, 0.7)

    def test_the_clean_pool_audit_interval_covers_the_failures_the_wires_missed(self):
        # the audit's claim is an interval, not a point: the true count must sit inside it
        true_missed = missed_by_tripwires(self.traces, self.report)
        audit = self.report.audit
        self.assertGreater(true_missed, 100)
        self.assertLessEqual(audit.lo * audit.population, true_missed)
        self.assertGreaterEqual(audit.hi * audit.population, true_missed)

    def test_estimates_are_in_the_right_ballpark(self):
        for mode in FAILURE_MODES:
            r = self.results[mode]
            self.assertLess(abs(r.est_failures - r.failing) / r.failing, 0.5, mode)

    def test_the_timeout_incident_has_a_changepoint_near_the_bad_deploy(self):
        change = incident_for(self.report, "tool_timeout_loop").changepoint
        self.assertIsNotNone(change)
        self.assertLess(abs(change.ts - 0.6 * len(self.traces)), 0.1 * len(self.traces))

    def test_the_timeout_contrast_points_at_the_failing_tool_span(self):
        contrast = incident_for(self.report, "tool_timeout_loop").contrast
        self.assertEqual(contrast.first_divergence, 1)

    def test_the_silent_incident_has_no_structural_divergence(self):
        contrast = incident_for(self.report, "wrong_customer_silent").contrast
        self.assertIsNone(contrast.first_divergence)

    def test_a_transient_error_that_retried_fine_is_not_an_incident(self):
        signatures = [i.estimate.key[1] for i in self.report.incidents]
        self.assertNotIn("<n> service unavailable", signatures)

    def test_the_brief_for_the_frontier_model_is_small(self):
        brief = incident_for(self.report, "bad_tool_args").brief()
        for key in ("cluster", "failure_class", "estimated_failures", "slices", "contrast", "example"):
            self.assertIn(key, brief)
        self.assertLess(estimate_tokens(json.dumps(brief)), 600)

    def test_incidents_are_ranked_by_estimated_failures(self):
        sizes = [i.estimate.est_failures for i in self.report.incidents]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_reports_render_as_text(self):
        text = format_report(self.report)
        for mode in FAILURE_MODES:
            self.assertIn(mode, text)
        self.assertIn("surfaced", format_evaluation(list(self.results.values()), self.report, self.traces))


class RunEdgeCaseTests(unittest.TestCase):
    def test_same_seed_gives_the_same_report(self):
        traces = generate(3000, seed=2)
        a = run(traces, make_judge(), seed=4)
        b = run(traces, make_judge(), seed=4)
        self.assertEqual(
            [(i.estimate.key, i.estimate.est_failures) for i in a.incidents],
            [(i.estimate.key, i.estimate.est_failures) for i in b.incidents],
        )

    def test_top_n_limits_the_incidents(self):
        report = run(generate(6000, seed=2), make_judge(), seed=1, top_n=2)
        self.assertLessEqual(len(report.incidents), 2)

    def test_no_traces_gives_an_empty_report(self):
        report = run([], make_judge())
        self.assertEqual((report.n_traces, report.judged, report.incidents), (0, 0, []))


if __name__ == "__main__":
    unittest.main()
