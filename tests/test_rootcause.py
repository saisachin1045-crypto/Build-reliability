import unittest

from tests.helpers import make_trace, span
from trace_triage.judge import Verdict
from trace_triage.rootcause import (
    changepoint_points,
    contrast_pair,
    find_changepoint,
    sample_points,
    slice_points,
    top_slices,
)
from trace_triage.sampling import JudgedSample


def sample(trace_id, failed, weight=1.0, ts=0, cls="c1", **attrs):
    trace = make_trace(trace_id, ts=ts, attrs=attrs)
    verdict = Verdict(not failed, cls if failed else None, None, 1.0, "t")
    return JudgedSample(trace, verdict, weight)


class TopSlicesTests(unittest.TestCase):
    def concentrated(self):
        out = []
        for i in range(20):
            model = "m-a" if i % 2 == 0 else "m-b"
            out.append(sample(f"x{i}", failed=i < 10, deploy="d42", model=model))
            out.append(sample(f"y{i}", failed=False, deploy="d41", model=model))
        return out

    def test_the_attribute_that_concentrates_failures_ranks_first(self):
        slices = top_slices(self.concentrated(), "c1", dims=("deploy", "model"), min_n=5)
        self.assertEqual((slices[0].dim, slices[0].value), ("deploy", "d42"))

    def test_when_two_slices_explain_the_same_failures_the_narrower_one_ranks_first(self):
        points = [sample(f"f{i}", True, deploy="d41", prompt_version="v36") for i in range(10)]
        points += [sample(f"a{i}", False, deploy="d41", prompt_version="v35") for i in range(20)]
        points += [sample(f"b{i}", False, deploy="d42", prompt_version="v35") for i in range(20)]
        points += [sample(f"c{i}", False, deploy="d42", prompt_version="v36") for i in range(5)]
        slices = top_slices(points, "c1", dims=("deploy", "prompt_version"), min_n=5)
        self.assertEqual((slices[0].dim, slices[0].value), ("prompt_version", "v36"))
        self.assertAlmostEqual(slices[0].excess, slices[1].excess)  # same failures explained, fewer traces

    def test_an_attribute_spread_evenly_over_failures_is_not_reported(self):
        slices = top_slices(self.concentrated(), "c1", dims=("model",), min_n=5)
        self.assertEqual(slices, [])

    def test_only_failures_of_the_requested_class_count(self):
        samples = [sample(f"a{i}", True, deploy="d42", cls="c1") for i in range(10)]
        samples += [sample(f"b{i}", True, deploy="d41", cls="c2") for i in range(10)]
        top = top_slices(samples, "c1", dims=("deploy",), min_n=5)[0]
        self.assertEqual((top.value, top.rate_in, top.rate_out), ("d42", 1.0, 0.0))

    def test_weights_scale_the_population_and_the_excess(self):
        samples = [sample(f"a{i}", failed=i < 5, weight=10, deploy="d42") for i in range(10)]
        samples += [sample(f"b{i}", failed=False, weight=90, deploy="d41") for i in range(10)]
        top = top_slices(samples, "c1", dims=("deploy",), min_n=5)[0]
        self.assertAlmostEqual(top.population, 100)
        self.assertAlmostEqual(top.rate_in, 0.5)
        self.assertAlmostEqual(top.excess, 50)

    def test_slices_with_too_few_samples_are_ignored(self):
        samples = [sample(f"a{i}", True, deploy="d42") for i in range(3)]
        samples += [sample(f"b{i}", False, deploy="d41") for i in range(20)]
        self.assertEqual(top_slices(samples, "c1", dims=("deploy",), min_n=5), [])

    def test_no_failures_means_no_slices(self):
        samples = [sample(f"a{i}", False, deploy="d42") for i in range(20)]
        self.assertEqual(top_slices(samples, "c1", dims=("deploy",), min_n=5), [])

    def test_a_missing_attribute_is_skipped_not_a_crash(self):
        samples = [sample(f"a{i}", i < 5, deploy="d42") for i in range(10)]
        self.assertEqual(top_slices(samples, "c1", dims=("tier",), min_n=2), [])


class PointFunctionTests(unittest.TestCase):
    """The same analyses on exact labels (every trace, weight 1), used when a tripwire already labels the cluster."""

    def test_changepoint_on_exact_labels_pins_the_onset(self):
        points = [(make_trace(f"t{ts}", ts=ts), ts >= 120 and ts % 3 == 0, 1.0) for ts in range(200)]
        self.assertEqual(changepoint_points(points).ts, 120)

    def test_slices_on_exact_labels(self):
        points = [(make_trace(f"t{i}", attrs={"deploy": "d42" if i < 20 else "d41"}), i < 10, 1.0) for i in range(40)]
        top = slice_points(points, dims=("deploy",), min_n=5)[0]
        self.assertEqual((top.value, top.rate_in, top.rate_out), ("d42", 0.5, 0.0))

    def test_sample_points_keep_weight_and_only_the_requested_class(self):
        samples = [sample("a", True, weight=7, cls="c1"), sample("b", True, weight=3, cls="c2"), sample("c", False)]
        points = sample_points(samples, "c1")
        self.assertEqual([(t.trace_id, failed, w) for t, failed, w in points], [("a", True, 7), ("b", False, 3), ("c", False, 1.0)])


class ChangepointTests(unittest.TestCase):
    def test_finds_the_time_the_failure_rate_jumped(self):
        samples = [sample(f"s{ts}", failed=(ts >= 120 and ts % 3 == 0), ts=ts) for ts in range(200)]
        change = find_changepoint(samples, "c1")
        self.assertIsNotNone(change)
        self.assertTrue(110 <= change.ts <= 130, change.ts)
        self.assertEqual(change.rate_before, 0.0)
        self.assertGreater(change.rate_after, 0.25)

    def test_a_steady_failure_rate_has_no_changepoint(self):
        samples = [sample(f"s{ts}", failed=ts % 5 == 0, ts=ts) for ts in range(200)]
        self.assertIsNone(find_changepoint(samples, "c1"))

    def test_a_drop_in_failures_is_not_reported_as_a_regression(self):
        samples = [sample(f"s{ts}", failed=(ts < 100 and ts % 3 == 0), ts=ts) for ts in range(200)]
        self.assertIsNone(find_changepoint(samples, "c1"))

    def test_too_little_data_returns_none(self):
        samples = [sample(f"s{ts}", failed=ts == 2, ts=ts) for ts in range(3)]
        self.assertIsNone(find_changepoint(samples, "c1"))


def lookup(trace_id, middle):
    return make_trace(trace_id, spans=[span("llm"), middle, span("llm")])


class ContrastTests(unittest.TestCase):
    def test_first_divergence_is_the_step_where_a_tool_started_failing(self):
        failing = lookup("f", span("tool", "get_order", status="timeout", error="slow"))
        passing = lookup("p", span("tool", "get_order"))
        result = contrast_pair(failing, [passing])
        self.assertEqual((result.passing_id, result.first_divergence), ("p", 1))
        self.assertIn("timeout", result.detail)
        self.assertIn("ok", result.detail)

    def test_picks_the_passing_trace_that_agrees_with_the_failing_one_longest(self):
        def run(trace_id, first_status, second_status):
            return make_trace(
                trace_id,
                spans=[
                    span("llm"),
                    span("tool", "a", status=first_status),
                    span("llm"),
                    span("tool", "b", status=second_status),
                ],
            )

        failing = run("f", "ok", "error")
        early = run("early", "error", "ok")  # diverges at span 1
        late = run("late", "ok", "ok")  # diverges at span 3
        self.assertEqual(contrast_pair(failing, [early, late]).passing_id, "late")

    def test_truncation_and_invalid_arguments_count_as_divergence(self):
        failing = make_trace("f", spans=[span("llm"), span("llm", finish_reason="length")])
        passing = make_trace("p", spans=[span("llm"), span("llm")])
        self.assertEqual(contrast_pair(failing, [passing]).first_divergence, 1)
        bad_args = make_trace("f2", spans=[span("tool", "book", args_valid=False)])
        good_args = make_trace("p2", spans=[span("tool", "book")])
        self.assertEqual(contrast_pair(bad_args, [good_args]).first_divergence, 0)

    def test_identical_structure_means_no_structural_divergence(self):
        failing = make_trace("f", spans=[span("llm"), span("tool", "get_order", args_hash="wrong"), span("llm")])
        passing = make_trace("p", spans=[span("llm"), span("tool", "get_order", args_hash="right"), span("llm")])
        result = contrast_pair(failing, [passing])
        self.assertIsNone(result.first_divergence)
        self.assertIn("no structural divergence", result.detail)

    def test_an_extra_trailing_span_is_a_divergence(self):
        failing = make_trace("f", spans=[span("llm"), span("tool", "t"), span("llm"), span("llm")])
        passing = make_trace("p", spans=[span("llm"), span("tool", "t"), span("llm")])
        result = contrast_pair(failing, [passing])
        self.assertEqual(result.first_divergence, 3)
        self.assertIn("<no span>", result.detail)

    def test_passing_traces_that_use_other_tools_are_not_comparable(self):
        failing = lookup("f", span("tool", "get_order", status="error", error="x"))
        other = lookup("p", span("tool", "book"))
        result = contrast_pair(failing, [other])
        self.assertIsNone(result.passing_id)
        self.assertIn("no passing trace", result.detail)

    def test_empty_pool(self):
        result = contrast_pair(lookup("f", span("tool", "t")), [])
        self.assertIsNone(result.passing_id)


if __name__ == "__main__":
    unittest.main()
