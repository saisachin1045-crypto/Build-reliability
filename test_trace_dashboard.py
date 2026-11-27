import http.client
import io
import json
import threading
import unittest
from unittest.mock import patch

from trace_dashboard import DashboardServer, MAX_LINE, records


class TraceUploadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = DashboardServer(("127.0.0.1", 0))
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join(timeout=5)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def upload(self, body, **headers):
        return self.request("POST", "/api/analyze", body, {"Content-Type": "application/x-ndjson", **headers})

    def test_dashboard_and_sample_report(self):
        status, page = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Open JSONL file", page)
        status, raw = self.request("GET", "/api/sample")
        self.assertEqual(status, 200)
        report = json.loads(raw)
        self.assertEqual(report["traces"], 1000)
        self.assertEqual(report["counts"]["unverified"], 800)

    def test_sample_download_round_trip_without_a_provider(self):
        status, sample = self.request("GET", "/sample.jsonl")
        self.assertEqual(status, 200)
        with patch("reliability.Model.request", side_effect=AssertionError("unexpected network use")):
            status, raw = self.upload(sample)
        self.assertEqual(status, 200)
        out = json.loads(raw)
        self.assertEqual(out["traces"], 20)
        self.assertEqual(out["counts"]["unverified"], 16)
        self.assertEqual(len(out["incidents"]), 4)
        self.assertEqual(out["usage"]["calls"], 0)

    def test_bad_line_does_not_produce_a_partial_report(self):
        data = b'{"trace_id":"one","user_goal":"find an order","final_answer":"ok"}\n{bad}\n'
        status, raw = self.upload(data)
        self.assertEqual(status, 400)
        self.assertIn("Line 2", json.loads(raw)["error"])
        status, _ = self.upload(b'{"trace_id":"a","final_answer":"ok","spans":[]}')
        self.assertEqual(status, 200)

    def test_invalid_shape_is_explained(self):
        for line in (b'null', b'{}', b'{"trace_id":"t","user_goal":[]}',
                     b'{"trace_id":"t","final_answer":"ok","spans":[false]}'):
            with self.subTest(line=line):
                status, raw = self.upload(line)
                self.assertEqual(status, 400)
                self.assertIn("Line 1", json.loads(raw)["error"])

    def test_reader_honors_length_and_line_limit(self):
        data = b'{"trace_id":"a","final_answer":"ok"}\n'
        stream = io.BytesIO(data + b'following request')
        self.assertEqual(next(iter(records(stream, len(data))))["trace_id"], "a")
        self.assertEqual(stream.read(), b'following request')
        with self.assertRaisesRegex(ValueError, "declared length"):
            list(records(io.BytesIO(data), len(data)+1))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            list(records(io.BytesIO(b'x'*(MAX_LINE+1)), MAX_LINE+1))

    def test_empty_wrong_origin_and_oversized_requests(self):
        self.assertEqual(self.upload(b'  \n')[0], 400)
        self.assertEqual(self.upload(b'{}', Origin="https://different.example")[0], 403)
        with patch("trace_dashboard.MAX_UPLOAD", 2):
            self.assertEqual(self.upload(b'123')[0], 413)

    def test_report_masks_email_in_evidence(self):
        trace = {"trace_id":"a","user_goal":"Where is order 12 for alice@example.test?",
                 "final_answer":"Order 13 shipped.","spans":[{"kind":"tool","status":"ok"}]}
        status, raw = self.upload(json.dumps(trace).encode())
        self.assertEqual(status, 200)
        self.assertNotIn(b"alice@example.test", raw)
        self.assertEqual(json.loads(raw)["incidents"][0]["failure_class"], "wrong_customer_silent")


if __name__ == "__main__":
    unittest.main()
