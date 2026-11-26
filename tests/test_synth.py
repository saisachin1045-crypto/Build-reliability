import unittest

from trace_triage.synth import DRIVERS, FAILURE_MODES, generate


class GenerateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.traces = generate(20000, seed=7)

    def test_same_seed_gives_identical_traces(self):
        self.assertEqual(generate(400, seed=3), generate(400, seed=3))

    def test_different_seed_gives_different_traces(self):
        self.assertNotEqual(generate(400, seed=3), generate(400, seed=4))

    def test_trace_ids_are_unique_and_timestamps_never_go_backwards(self):
        ids = [t.trace_id for t in self.traces]
        self.assertEqual(len(ids), len(set(ids)))
        stamps = [t.ts for t in self.traces]
        self.assertEqual(stamps, sorted(stamps))

    def test_every_seeded_failure_mode_appears(self):
        modes = {t.truth.failure_mode for t in self.traces if t.truth.failure_mode}
        self.assertEqual(modes, set(FAILURE_MODES))

    def test_overall_failure_rate_is_a_few_percent(self):
        failing = sum(1 for t in self.traces if t.truth.failure_mode)
        self.assertTrue(0.04 < failing / len(self.traces) < 0.14, failing)

    def test_failing_traces_carry_the_attribute_that_drives_their_mode(self):
        for t in self.traces:
            mode = t.truth.failure_mode
            if mode:
                dim, value = DRIVERS[mode]
                self.assertEqual(t.attrs[dim], value, mode)

    def test_silent_mode_is_silent_every_span_ok_and_an_answer_exists(self):
        silent = [t for t in self.traces if t.truth.failure_mode == "wrong_customer_silent"]
        self.assertGreater(len(silent), 100)
        for t in silent:
            self.assertTrue(all(s.status == "ok" for s in t.spans))
            self.assertIsNotNone(t.final_answer)

    def test_loud_modes_leave_a_structural_mark(self):
        def of(mode):
            return [t for t in self.traces if t.truth.failure_mode == mode]

        self.assertTrue(all(any(s.status == "timeout" for s in t.spans) for t in of("tool_timeout_loop")))
        self.assertTrue(all(any(not s.args_valid for s in t.spans) for t in of("bad_tool_args")))
        self.assertTrue(
            all(any(s.finish_reason == "length" for s in t.spans) for t in of("context_truncated"))
        )

    def test_timeout_loop_starts_only_after_the_bad_deploy(self):
        stamps = [t.ts for t in self.traces if t.truth.failure_mode == "tool_timeout_loop"]
        self.assertGreater(min(stamps), 0.55 * len(self.traces))

    def test_some_healthy_traces_hit_a_transient_error_and_still_answer(self):
        transient = [
            t
            for t in self.traces
            if not t.truth.failure_mode and any(s.status == "error" for s in t.spans)
        ]
        self.assertGreater(len(transient), 20)
        self.assertTrue(all(t.final_answer is not None for t in transient))

    def test_retries_repeat_the_previous_goal_inside_a_session(self):
        by_session = {}
        for t in self.traces:
            by_session.setdefault(t.session_id, []).append(t)
        repeats = sum(
            1
            for group in by_session.values()
            for a, b in zip(group, group[1:])
            if a.user_goal == b.user_goal
        )
        self.assertGreater(repeats, 50)


if __name__ == "__main__":
    unittest.main()
