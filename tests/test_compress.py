import unittest

from tests.helpers import failing_truth, make_trace, span
from trace_triage.compress import compress, estimate_tokens, head_tail, redact


class RedactTests(unittest.TestCase):
    def test_emails_are_redacted(self):
        self.assertEqual(redact("mail me at jo.doe@example.com now"), "mail me at <email> now")

    def test_long_digit_runs_like_phone_numbers_are_redacted(self):
        self.assertEqual(redact("call +91 98765 43210 please"), "call <phone> please")

    def test_short_ids_such_as_order_numbers_are_kept(self):
        self.assertEqual(redact("Where is order 4821? slot 12 4821"), "Where is order 4821? slot 12 4821")


class HeadTailTests(unittest.TestCase):
    def test_short_text_is_unchanged(self):
        self.assertEqual(head_tail("short", 20), "short")

    def test_long_text_keeps_head_and_tail_and_says_how_much_was_cut(self):
        text = "A" * 10 + "m" * 80 + "Z" * 10
        out = head_tail(text, 20)
        self.assertTrue(out.startswith("A" * 10))
        self.assertTrue(out.endswith("Z" * 10))
        self.assertIn("[...80 chars...]", out)
        self.assertTrue(out.isascii())


class EstimateTokensTests(unittest.TestCase):
    def test_roughly_four_characters_per_token(self):
        self.assertEqual(estimate_tokens("x" * 400), 100)


class CompressTests(unittest.TestCase):
    def looped(self, **kw):
        spans = (
            [span("llm", tokens_in=1800, tokens_out=150)]
            + [
                span("tool", "get_order", status="timeout", error="timeout after 30000ms", args_hash="h")
                for _ in range(4)
            ]
            + [span("llm")]
        )
        return make_trace(spans=spans, goal="Where is my order 4821?", final=None, **kw)

    def test_repeated_identical_spans_collapse_with_a_count_and_keep_their_index(self):
        text = compress(self.looped())
        self.assertIn("[1] tool:get_order timeout x4", text)
        self.assertIn('err="timeout after <n>ms"', text)

    def test_a_long_run_of_spans_compresses_to_few_lines(self):
        spans = [span("tool", "poll", args_hash="same") for _ in range(50)]
        self.assertLess(len(compress(make_trace(spans=spans)).splitlines()), 6)

    def test_missing_final_answer_and_user_signal_are_shown(self):
        text = compress(self.looped(user_signal="thumbs_down"))
        self.assertIn("final: <none>", text)
        self.assertIn("signal: thumbs_down", text)

    def test_goal_and_answer_are_redacted(self):
        trace = make_trace(goal="email jo@example.com", final="sent to jo@example.com")
        text = compress(trace)
        self.assertNotIn("jo@example.com", text)
        self.assertIn("<email>", text)

    def test_a_long_answer_is_cut_to_head_and_tail(self):
        text = compress(make_trace(final="x" * 5000))
        self.assertLess(len(text), 600)

    def test_ground_truth_never_leaks_into_what_the_judge_reads(self):
        plain = compress(make_trace())
        labelled = compress(make_trace(truth=failing_truth("bad_tool_args")))
        self.assertEqual(plain, labelled)


if __name__ == "__main__":
    unittest.main()
