import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from reliability import Model, analyze, classify, dashboard, demo, read_jsonl, redact, replay

class ReliabilityTests(unittest.TestCase):
    def test_demo_counts(self):
        r=analyze(demo(1000))
        self.assertEqual(r["traces"],1000)
        self.assertEqual(r["counts"]["unverified"],800)
        self.assertEqual(len(r["incidents"]),4)
        self.assertEqual(r["usage"]["calls"],0)
        self.assertEqual(sum(r["counts"].values()),1000)
        self.assertTrue(all(i["count"]==50 for i in r["incidents"]))

    def test_truth_is_ignored(self):
        t=next(demo(1,True)); t["truth"]={"failure_mode":"fake"}
        self.assertEqual(classify(t)[0],"unverified")

    def test_recovered_timeout(self):
        t=next(demo(1,True))
        t["spans"]=[{"kind":"tool","status":"timeout"}]*3+[{"kind":"tool","status":"ok"}]
        self.assertEqual(classify(t)[0],"unverified")

    def test_replay(self):
        r=replay(demo(100),demo(100,True))
        self.assertEqual(r["signals_removed"],20)
        self.assertEqual(r["remaining_signals"],0)
        self.assertEqual(r["matched"],100)

    def test_redaction(self):
        r=redact({"email":"person@example.com","password":"example","note":"Bearer fake-token","ssn":"123-45-6789"})
        self.assertNotIn("person@example.com",str(r))
        self.assertNotIn("fake-token",str(r))
        self.assertNotIn("123-45-6789",str(r))
        self.assertEqual(r["password"],"[secret]")

    def test_jsonl_dashboard(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"traces.jsonl"
            path.write_text("\n".join(json.dumps(t) for t in demo(20)))
            r=analyze(read_jsonl(path))
            target=Path(d)/"report.html"
            r["incidents"][0]["example"]={"x":"<script>alert(1)</script>"}
            dashboard(r,target)
            self.assertIn("Filter incidents",target.read_text())
            self.assertNotIn("<script>alert(1)</script>",target.read_text())

    def test_ollama_request_and_cache(self):
        m=Model("ollama",budget=1)
        reply={"message":{"content":"{\"verdict\":\"pass\"}"},"prompt_eval_count":10,"eval_count":5}
        with patch("reliability.urllib.request.urlopen") as url:
            url.return_value.__enter__.return_value.read.return_value=json.dumps(reply).encode()
            self.assertEqual(m.ask("judge",{"text":"x"}),{"verdict":"pass"})
            self.assertEqual(m.ask("judge",{"text":"x"}),{"verdict":"pass"})
            self.assertEqual(m.calls,1)
            self.assertEqual(m.tokens_in,10)
            self.assertRaises(RuntimeError,m.ask,"different",{})

    def test_tool_loop_and_embeddings(self):
        m=Model("ollama",budget=12)
        def reply(url,payload,headers=None):
            m.calls+=1
            if url.endswith("/embed"):
                return {"embeddings":[[1,0]]*4}
            prompt=payload["messages"][0]["content"]
            result={"tool":"compare_deployments"} if "Choose one tool" in prompt else {"hypothesis":"Check deployment"}
            return {"message":{"content":json.dumps(result)},"prompt_eval_count":2,"eval_count":1}
        with patch.object(m,"request",side_effect=reply):
            r=analyze(demo(100),m,embeddings=True)
        self.assertTrue(all("model_diagnosis" in i for i in r["incidents"]))
        self.assertEqual(len(r["semantic_cluster_links"]),6)
        self.assertLessEqual(m.calls,12)

    def test_empty(self):
        self.assertEqual(analyze([])["traces"],0)

    def test_model_error_is_reported(self):
        m=Model("ollama",budget=2)
        with patch("reliability.urllib.request.urlopen",side_effect=OSError("offline")):
            r=analyze(demo(20),m)
        self.assertTrue(r["model_errors"])
        self.assertEqual(r["traces"],20)

if __name__=="__main__": unittest.main()
