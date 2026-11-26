import random
import unittest

from tests.helpers import make_trace
from trace_triage.judge import Verdict
from trace_triage.sampling import (
    Stratum,
    audit_clean_pool,
    estimate_cluster,
    stratified_estimate,
    wilson_interval,
)


class FakeJudge:
    """Fails exactly the trace ids it is told to fail; counts calls."""

    tier = "fake"

    def __init__(self, failing_ids=()):
        self.failing = set(failing_ids)
        self.calls = 0

    def judge(self, trace):
        self.calls += 1
        failed = trace.trace_id in self.failing
        return Verdict(not failed, "x" if failed else None, None, 1.0, "fake")


def traces(prefix, count):
    return [make_trace(f"{prefix}{i}") for i in range(count)]


class WilsonTests(unittest.TestCase):
    def test_zero_of_ten_still_leaves_room_for_failures(self):
        lo, hi = wilson_interval(0, 10)
        self.assertAlmostEqual(lo, 0.0, places=9)
        self.assertTrue(0.25 < hi < 0.30, hi)

    def test_all_failing_has_a_high_lower_bound(self):
        lo, hi = wilson_interval(10, 10)
        self.assertGreater(lo, 0.7)
        self.assertAlmostEqual(hi, 1.0, places=9)

    def test_no_samples_means_total_uncertainty(self):
        self.assertEqual(wilson_interval(0, 0), (0.0, 1.0))

    def test_interval_narrows_as_the_sample_grows(self):
        small = wilson_interval(5, 10)
        large = wilson_interval(50, 100)
        self.assertLess(large[1] - large[0], small[1] - small[0])


class StratifiedEstimateTests(unittest.TestCase):
    def test_oversampling_a_failing_stratum_does_not_bias_the_estimate(self):
        strata = [Stratum(size=100, sampled=10, failures=10), Stratum(size=900, sampled=10, failures=0)]
        rate, lo, hi = stratified_estimate(strata)
        self.assertAlmostEqual(rate, 0.1)  # a naive pooled rate would be 0.5
        self.assertLessEqual(lo, rate)
        self.assertGreaterEqual(hi, rate)

    def test_a_census_has_no_uncertainty(self):
        rate, lo, hi = stratified_estimate([Stratum(size=10, sampled=10, failures=3)])
        self.assertEqual((rate, lo, hi), (0.3, 0.3, 0.3))

    def test_no_strata_gives_zero(self):
        self.assertEqual(stratified_estimate([]), (0.0, 0.0, 0.0))

    def test_a_non_empty_stratum_with_no_samples_is_an_error(self):
        with self.assertRaises(ValueError):
            stratified_estimate([Stratum(size=10, sampled=0, failures=0)])


class EstimateClusterTests(unittest.TestCase):
    def estimate(self, members, flags, judge, seed=1, **kw):
        return estimate_cluster(("shape", "-"), members, flags, judge, random.Random(seed), **kw)

    def test_a_small_cluster_is_judged_in_full_and_exact(self):
        members = traces("a", 6)
        judge = FakeJudge({"a0", "a1"})
        est, _ = self.estimate(members, {t.trace_id: frozenset() for t in members}, judge)
        self.assertEqual(est.judged, 6)
        self.assertAlmostEqual(est.rate, 2 / 6)
        self.assertEqual(est.lo, est.hi)

    def test_flagged_oversampling_is_corrected_by_stratum_weights(self):
        flagged, clean = traces("f", 20), traces("c", 180)
        flags = {t.trace_id: frozenset({"user_thumbs_down"}) for t in flagged}
        flags.update({t.trace_id: frozenset() for t in clean})
        judge = FakeJudge({t.trace_id for t in flagged})  # every flagged trace failed, no clean one did
        est, _ = self.estimate(flagged + clean, flags, judge)
        self.assertAlmostEqual(est.rate, 20 / 200)
        self.assertAlmostEqual(est.est_failures, 20)

    def test_a_clean_stratum_with_no_failures_keeps_widening(self):
        clean = traces("c", 500)
        est, _ = self.estimate(clean, {t.trace_id: frozenset() for t in clean}, FakeJudge())
        self.assertGreater(est.judged, 10)

    def test_widening_stops_at_the_cap(self):
        clean = traces("c", 500)
        judge = FakeJudge()
        self.estimate(clean, {t.trace_id: frozenset() for t in clean}, judge, base=10, cap=20)
        self.assertEqual(judge.calls, 20)

    def test_a_clearly_failing_stratum_stops_early(self):
        flagged = traces("f", 500)
        flags = {t.trace_id: frozenset({"loop"}) for t in flagged}
        judge = FakeJudge({t.trace_id for t in flagged})
        est, _ = self.estimate(flagged, flags, judge, base=10)
        self.assertEqual(est.judged, 10)
        self.assertGreater(est.rate, 0.99)

    def test_failure_classes_are_tallied_from_the_verdicts(self):
        members = traces("a", 6)
        judge = FakeJudge({"a0", "a1", "a2"})
        est, _ = self.estimate(members, {t.trace_id: frozenset() for t in members}, judge)
        self.assertEqual(est.classes, {"x": 3})

    def test_a_failure_with_no_class_is_tallied_as_unclassified(self):
        class Unclassifying(FakeJudge):
            def judge(self, trace):
                return Verdict(False, None, None, 1.0, "fake")

        members = traces("a", 4)
        est, _ = self.estimate(members, {t.trace_id: frozenset() for t in members}, Unclassifying())
        self.assertEqual(est.classes, {"unclassified": 4})

    def test_sample_weights_add_up_to_the_cluster_size(self):
        flagged, clean = traces("f", 40), traces("c", 300)
        flags = {t.trace_id: frozenset({"retry"}) for t in flagged}
        flags.update({t.trace_id: frozenset() for t in clean})
        _, samples = self.estimate(flagged + clean, flags, FakeJudge({"f0"}))
        self.assertAlmostEqual(sum(s.weight for s in samples), 340)

    def test_the_same_seed_draws_the_same_sample(self):
        members = traces("a", 300)
        flags = {t.trace_id: frozenset() for t in members}
        _, one = self.estimate(members, flags, FakeJudge(), seed=5)
        _, two = self.estimate(members, flags, FakeJudge(), seed=5)
        self.assertEqual([s.trace.trace_id for s in one], [s.trace.trace_id for s in two])


class AuditTests(unittest.TestCase):
    def test_estimates_what_the_tripwires_missed_from_a_random_sample(self):
        pool = traces("p", 1000)
        judge = FakeJudge({t.trace_id for t in pool[:100]})  # 10% of the clean pool failed silently
        result = audit_clean_pool(pool, judge, random.Random(0), n=300)
        self.assertEqual(result.judged, 300)
        self.assertLess(abs(result.rate - 0.1), 0.05)
        self.assertLess(abs(result.est_missed - 100), 50)

    def test_a_small_pool_is_judged_in_full(self):
        pool = traces("p", 5)
        result = audit_clean_pool(pool, FakeJudge({"p0"}), random.Random(0), n=300)
        self.assertEqual((result.judged, result.failures), (5, 1))
        self.assertAlmostEqual(result.rate, 0.2)

    def test_an_empty_pool_reports_zero(self):
        result = audit_clean_pool([], FakeJudge(), random.Random(0), n=300)
        self.assertEqual((result.population, result.judged, result.est_missed), (0, 0, 0.0))


if __name__ == "__main__":
    unittest.main()
