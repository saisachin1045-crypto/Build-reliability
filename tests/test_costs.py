import unittest

from trace_triage.costs import (
    Assumptions,
    estimate,
    format_table,
    naive_cost,
    stage3_cost,
    total,
)


class NaiveCostTests(unittest.TestCase):
    def test_judging_every_raw_trace_on_sonnet_costs_120k(self):
        # 1M traces x 60k tokens = 60B tokens, at $2 per 1M input tokens
        self.assertAlmostEqual(naive_cost(Assumptions(), "sonnet-5.5"), 120_000.0)

    def test_batch_halves_it(self):
        self.assertAlmostEqual(naive_cost(Assumptions(), "sonnet-5.5", batch=True), 60_000.0)

    def test_other_tiers_scale_with_their_input_price(self):
        self.assertAlmostEqual(naive_cost(Assumptions(), "haiku-5.5"), 6_000.0)
        self.assertAlmostEqual(naive_cost(Assumptions(), "opus-5.5"), 240_000.0)

    def test_unknown_tier_is_an_error(self):
        with self.assertRaises(KeyError):
            naive_cost(Assumptions(), "gpt-nine")


class Stage3Tests(unittest.TestCase):
    def test_judged_count_is_cluster_samples_plus_the_clean_audit(self):
        self.assertEqual(Assumptions().judged, 22_000)

    def test_sonnet_on_batch_matches_the_hand_calculation(self):
        # 22,000 x 4,000 = 88M tokens in at $2 = $176; 22,000 x 300 = 6.6M out at $10 = $66; batch halves it
        self.assertAlmostEqual(stage3_cost(Assumptions(), "sonnet-5.5", batch=True), 121.0)

    def test_without_batch_it_costs_double(self):
        a = Assumptions()
        self.assertAlmostEqual(stage3_cost(a, "sonnet-5.5", batch=False), 2 * stage3_cost(a, "sonnet-5.5", batch=True))


class EstimateTests(unittest.TestCase):
    def test_every_stage_has_a_low_not_above_its_high(self):
        for stage in estimate():
            self.assertLessEqual(stage.low, stage.high, stage.name)

    def test_total_is_the_sum_of_the_stages(self):
        stages = estimate()
        t = total(stages)
        self.assertAlmostEqual(t.low, sum(s.low for s in stages))
        self.assertAlmostEqual(t.high, sum(s.high for s in stages))

    def test_default_total_matches_the_range_quoted_in_the_answer(self):
        t = total(estimate())
        self.assertAlmostEqual(t.low, 351, delta=1)
        self.assertAlmostEqual(t.high, 827, delta=1)

    def test_the_funnel_is_far_cheaper_than_judging_everything_even_on_batch(self):
        t = total(estimate())
        self.assertGreater(naive_cost(Assumptions(), "sonnet-5.5", batch=True) / t.high, 50)

    def test_fewer_clusters_cost_less_to_judge(self):
        small = total(estimate(Assumptions(clusters=500)))
        self.assertLess(small.high, total(estimate()).high)


class FormatTests(unittest.TestCase):
    def test_table_lists_each_stage_the_total_and_the_naive_comparison(self):
        text = format_table(Assumptions())
        for fragment in ("1 tripwires", "2 embeddings", "3 sampled judging", "4 root cause", "5 verify", "TOTAL", "judging every raw trace"):
            self.assertIn(fragment, text)


if __name__ == "__main__":
    unittest.main()
