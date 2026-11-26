import unittest
from unittest.mock import Mock
from reliability import Model
from tests.helpers import make_trace, failing_truth
from trace_triage.live import ModelJudge, ObservableJudge, analyze, observable
from trace_triage.synth import generate


class IntegratedTests(unittest.TestCase):
    def test_no_truth_in_judge_payload(self):
        trace = make_trace(truth=failing_truth())
        self.assertNotIn("truth", observable(trace))
        self.assertTrue(ObservableJudge().judge(trace).goal_met)

    def test_real_model_adapter(self):
        model = Mock(name="provider")
        model.name = "local-model"
        model.ask.return_value = {"goal_met": False, "failure_class": "other", "first_bad_span": 1, "confidence": 0.9}
        self.assertFalse(ModelJudge(model).judge(make_trace()).goal_met)
        self.assertNotIn("truth", model.ask.call_args.args[1])

    def test_invalid_model_verdict_is_rejected(self):
        model = Mock()
        model.ask.return_value = {"goal_met": "false", "confidence": 1}
        with self.assertRaises(ValueError):
            ModelJudge(model).judge(make_trace())

    def test_original_funnel_runs_without_keys(self):
        report = analyze(generate(1000), Model())
        self.assertEqual(report["traces"], 1000)
        self.assertGreater(report["judged_samples"], 0)
        self.assertEqual(report["usage"]["calls"], 0)
        self.assertTrue(report["incidents"])
