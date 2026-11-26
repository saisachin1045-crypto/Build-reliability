import unittest

from tests.helpers import make_trace, span
from trace_triage.clustering import cluster, cluster_key
from trace_triage.fingerprint import shape_key


def booking(trace_id, error):
    return make_trace(
        trace_id, spans=[span("llm"), span("tool", "book", status="error", error=error)]
    )


class ClusterKeyTests(unittest.TestCase):
    def test_clean_trace_gets_a_dash_signature(self):
        trace = make_trace()
        self.assertEqual(cluster_key(trace, frozenset()), (shape_key(trace), "-"))

    def test_different_error_text_splits_clusters_with_the_same_shape(self):
        a = cluster_key(booking("a", "422 invalid slot"), frozenset({"error_span"}))
        b = cluster_key(booking("b", "503 unavailable"), frozenset({"error_span"}))
        self.assertEqual(a[0], b[0])
        self.assertNotEqual(a, b)

    def test_same_error_with_different_numbers_shares_a_cluster(self):
        a = cluster_key(booking("a", "422 invalid slot_id: 1"), frozenset({"error_span"}))
        b = cluster_key(booking("b", "422 invalid slot_id: 999"), frozenset({"error_span"}))
        self.assertEqual(a, b)

    def test_loud_flag_names_the_cluster_when_there_is_no_error_text(self):
        trace = make_trace()
        key = cluster_key(trace, frozenset({"truncated", "token_outlier"}))
        self.assertEqual(key[1], "flag:truncated")

    def test_weak_flags_do_not_split_a_cluster(self):
        trace = make_trace()
        weak = frozenset({"user_thumbs_down", "token_outlier", "latency_outlier"})
        self.assertEqual(cluster_key(trace, weak)[1], "-")


class ClusterTests(unittest.TestCase):
    def test_every_trace_lands_in_exactly_one_cluster(self):
        traces = [make_trace("a"), make_trace("b"), booking("c", "422 x"), booking("d", "503 y")]
        flags = {t.trace_id: frozenset() for t in traces}
        clusters = cluster(traces, flags)
        members = [t.trace_id for group in clusters.values() for t in group]
        self.assertEqual(sorted(members), ["a", "b", "c", "d"])
        self.assertEqual(len(clusters), 3)


if __name__ == "__main__":
    unittest.main()
