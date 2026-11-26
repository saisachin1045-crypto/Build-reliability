import unittest

from tests.helpers import make_trace
from trace_triage.sessions import join_sessions


class JoinSessionsTests(unittest.TestCase):
    def test_marks_rephrase_when_the_next_trace_in_the_session_repeats_the_goal(self):
        first = make_trace("a", session_id="s1", ts=1, goal="Where is order 12?")
        second = make_trace("b", session_id="s1", ts=2, goal="where is order 12 ?")
        out = join_sessions([first, second])
        self.assertEqual(out[0].user_signal, "rephrase")
        self.assertIsNone(out[1].user_signal)

    def test_does_not_mark_when_the_next_goal_is_different(self):
        out = join_sessions(
            [
                make_trace("a", session_id="s1", goal="Where is order 12?"),
                make_trace("b", session_id="s1", goal="Book slot 4"),
            ]
        )
        self.assertIsNone(out[0].user_signal)

    def test_never_joins_across_sessions(self):
        out = join_sessions(
            [
                make_trace("a", session_id="s1", goal="Where is order 12?"),
                make_trace("b", session_id="s2", goal="Where is order 12?"),
            ]
        )
        self.assertIsNone(out[0].user_signal)

    def test_keeps_an_existing_user_signal(self):
        out = join_sessions(
            [
                make_trace("a", session_id="s1", goal="same", user_signal="thumbs_down"),
                make_trace("b", session_id="s1", goal="same"),
            ]
        )
        self.assertEqual(out[0].user_signal, "thumbs_down")

    def test_does_not_modify_the_input_traces(self):
        first = make_trace("a", session_id="s1", goal="same")
        second = make_trace("b", session_id="s1", goal="same")
        join_sessions([first, second])
        self.assertIsNone(first.user_signal)


if __name__ == "__main__":
    unittest.main()
