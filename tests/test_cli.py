import contextlib
import io
import unittest

from trace_triage.__main__ import main


def run_cli(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = main(list(argv))
    return code, out.getvalue()


class CliTests(unittest.TestCase):
    def test_cost_prints_the_funnel_and_the_naive_comparison(self):
        code, text = run_cli("cost")
        self.assertEqual(code, 0)
        self.assertIn("TOTAL", text)
        self.assertIn("judging every raw trace", text)

    def test_demo_runs_the_pipeline_and_prints_incidents_and_the_evaluation(self):
        code, text = run_cli("demo", "--n", "4000", "--seed", "3")
        self.assertEqual(code, 0)
        self.assertIn("Incidents", text)
        self.assertIn("seeded failure modes", text)
        self.assertIn("Judge calibration", text)

    def test_sweep_reports_how_often_the_pipeline_gets_every_mode_right(self):
        code, text = run_cli("sweep", "--seeds", "2", "--n", "4000")
        self.assertEqual(code, 0)
        self.assertIn("seeds", text)

    def test_an_unknown_command_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(["nope"])


if __name__ == "__main__":
    unittest.main()
