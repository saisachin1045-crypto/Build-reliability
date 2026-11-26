import unittest

from tests.helpers import failing_truth, make_trace, span
from trace_triage.fingerprint import shape_key
from trace_triage.tripwires import ShapeStats, compute_flags


def flags_of(trace, stats=None):
    return compute_flags(trace, stats or ShapeStats.fit([trace]))


def llm_only(trace_id, tokens=1000, latency=500):
    return make_trace(trace_id, spans=[span("llm", tokens_in=tokens, latency_ms=latency)])


class FlagTests(unittest.TestCase):
    def test_clean_trace_has_no_flags(self):
        self.assertEqual(flags_of(make_trace()), frozenset())

    def test_error_span(self):
        trace = make_trace(spans=[span("tool", "t", status="error", error="boom")])
        self.assertEqual(flags_of(trace), {"error_span"})

    def test_timeout_is_also_an_error_span(self):
        trace = make_trace(spans=[span("tool", "t", status="timeout", error="slow")])
        self.assertEqual(flags_of(trace), {"error_span", "timeout"})

    def test_two_identical_tool_calls_are_a_retry_not_a_loop(self):
        trace = make_trace(spans=[span("tool", "t", args_hash="x"), span("tool", "t", args_hash="x")])
        self.assertEqual(flags_of(trace), {"retry"})

    def test_three_identical_tool_calls_are_a_loop(self):
        trace = make_trace(spans=[span("tool", "t", args_hash="x") for _ in range(3)])
        self.assertEqual(flags_of(trace), {"retry", "loop"})

    def test_same_tool_with_different_arguments_is_not_a_retry(self):
        trace = make_trace(spans=[span("tool", "t", args_hash="x"), span("tool", "t", args_hash="y")])
        self.assertEqual(flags_of(trace), frozenset())

    def test_invalid_tool_arguments(self):
        trace = make_trace(spans=[span("tool", "t", args_valid=False)])
        self.assertEqual(flags_of(trace), {"invalid_args"})

    def test_truncated_output(self):
        trace = make_trace(spans=[span("llm", finish_reason="length")])
        self.assertEqual(flags_of(trace), {"truncated"})

    def test_missing_final_answer(self):
        self.assertEqual(flags_of(make_trace(final=None)), {"no_final_answer"})

    def test_user_signals_become_flags(self):
        expected = {
            "thumbs_down": "user_thumbs_down",
            "rephrase": "user_rephrase",
            "escalated": "user_escalated",
            "abandoned": "user_abandoned",
        }
        for signal, flag in expected.items():
            self.assertEqual(flags_of(make_trace(user_signal=signal)), {flag}, signal)

    def test_flags_never_read_ground_truth(self):
        trace = make_trace(truth=failing_truth())  # spans are clean, truth says failing
        self.assertEqual(flags_of(trace), frozenset())


class OutlierTests(unittest.TestCase):
    def setUp(self):
        self.normal = [llm_only(f"n{i}") for i in range(9)]

    def test_token_outlier_is_relative_to_the_shape_median(self):
        big, medium = llm_only("big", tokens=4000), llm_only("medium", tokens=2500)
        stats = ShapeStats.fit(self.normal + [big])
        self.assertIn("token_outlier", compute_flags(big, stats))
        self.assertNotIn("token_outlier", compute_flags(medium, stats))

    def test_latency_outlier_is_relative_to_the_shape_median(self):
        slow = llm_only("slow", latency=2000)
        stats = ShapeStats.fit(self.normal + [slow])
        self.assertIn("latency_outlier", compute_flags(slow, stats))

    def test_a_shape_never_seen_in_fit_cannot_be_an_outlier(self):
        stats = ShapeStats.fit(self.normal)
        odd = make_trace("odd", spans=[span("tool", "other", latency_ms=10**6)])
        self.assertNotIn("latency_outlier", compute_flags(odd, stats))


class ShapeStatsTests(unittest.TestCase):
    def test_median_tokens_and_latency_per_shape(self):
        traces = [llm_only("a", tokens=100, latency=10), llm_only("b", tokens=300, latency=30), llm_only("c", tokens=200, latency=20)]
        stats = ShapeStats.fit(traces)
        key = shape_key(traces[0])
        self.assertEqual(stats.tokens[key], 200)
        self.assertEqual(stats.latency[key], 20)


if __name__ == "__main__":
    unittest.main()
