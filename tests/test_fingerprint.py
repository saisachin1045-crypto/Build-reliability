import unittest

from tests.helpers import make_trace, span
from trace_triage.fingerprint import error_signature, normalise_error, shape_key, workflow_key


class NormaliseErrorTests(unittest.TestCase):
    def test_numbers_become_placeholders(self):
        self.assertEqual(normalise_error("422 invalid slot_id: 8841"), "<n> invalid slot_id: <n>")

    def test_numbers_glued_to_units_are_normalised(self):
        self.assertEqual(
            normalise_error("timeout after 30000ms calling orders-api"),
            "timeout after <n>ms calling orders-api",
        )

    def test_uuids_and_long_hex_ids_become_id(self):
        message = "row 3f2b8c1e-9a4d-4e0b-8c11-0d2f5a6b7c8d locked by deadbeefcafe1234"
        self.assertEqual(normalise_error(message), "row <id> locked by <id>")

    def test_quoted_strings_become_s(self):
        self.assertEqual(normalise_error("unknown tool 'get_ordr'"), "unknown tool <s>")

    def test_same_error_with_different_ids_has_the_same_signature(self):
        self.assertEqual(
            normalise_error("422 invalid slot_id: 1"), normalise_error("422 invalid slot_id: 99999")
        )


class ShapeKeyTests(unittest.TestCase):
    def test_model_name_does_not_change_the_shape(self):
        a = make_trace(spans=[span("llm", "m-large"), span("tool", "get_order")])
        b = make_trace(spans=[span("llm", "m-mid"), span("tool", "get_order")])
        self.assertEqual(shape_key(a), shape_key(b))

    def test_single_call_is_not_marked_as_a_loop(self):
        trace = make_trace(spans=[span("llm"), span("tool", "get_order"), span("llm")])
        self.assertEqual(shape_key(trace), "llm>tool:get_order>llm")

    def test_repeated_consecutive_tool_calls_collapse_to_a_loop_marker(self):
        spans = [span("llm")] + [span("tool", "get_order") for _ in range(4)] + [span("llm")]
        self.assertEqual(shape_key(make_trace(spans=spans)), "llm>tool:get_order*>llm")

    def test_two_repeats_and_five_repeats_share_a_shape(self):
        two = make_trace(spans=[span("tool", "t"), span("tool", "t")])
        five = make_trace(spans=[span("tool", "t") for _ in range(5)])
        self.assertEqual(shape_key(two), shape_key(five))

    def test_different_tool_order_gives_a_different_shape(self):
        a = make_trace(spans=[span("tool", "a"), span("tool", "b")])
        b = make_trace(spans=[span("tool", "b"), span("tool", "a")])
        self.assertNotEqual(shape_key(a), shape_key(b))


class WorkflowKeyTests(unittest.TestCase):
    def test_order_and_repeats_do_not_matter(self):
        a = make_trace(spans=[span("llm"), span("tool", "b"), span("tool", "a"), span("tool", "a")])
        b = make_trace(spans=[span("llm"), span("tool", "a"), span("tool", "b")])
        self.assertEqual(workflow_key(a), workflow_key(b))

    def test_a_loop_and_a_single_call_share_a_workflow(self):
        one = make_trace(spans=[span("llm"), span("tool", "get_order"), span("llm")])
        four = make_trace(spans=[span("llm")] + [span("tool", "get_order") for _ in range(4)] + [span("llm")])
        self.assertEqual(workflow_key(one), workflow_key(four))
        self.assertNotEqual(shape_key(one), shape_key(four))

    def test_different_tools_are_different_workflows(self):
        self.assertNotEqual(
            workflow_key(make_trace(spans=[span("tool", "a")])),
            workflow_key(make_trace(spans=[span("retrieval", "a")])),
        )

    def test_llm_only_traces_share_one_key(self):
        self.assertEqual(workflow_key(make_trace(spans=[span("llm")])), "llm-only")


class ErrorSignatureTests(unittest.TestCase):
    def test_clean_trace_has_no_signature(self):
        self.assertIsNone(error_signature(make_trace()))

    def test_uses_the_first_failing_span_and_normalises_it(self):
        spans = [
            span("llm"),
            span("tool", "book", status="error", error="422 invalid slot_id: 77"),
            span("tool", "book", status="error", error="other"),
        ]
        self.assertEqual(error_signature(make_trace(spans=spans)), "<n> invalid slot_id: <n>")

    def test_failing_span_without_a_message_gives_no_signature(self):
        trace = make_trace(spans=[span("tool", "x", status="error", error=None)])
        self.assertIsNone(error_signature(trace))


if __name__ == "__main__":
    unittest.main()
