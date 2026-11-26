# Verified local execution

Source blob: `c89b835a4744b156e9d1b7600e4a0d00ff1873d7` (local copy and GitHub blob matched).
Python 3.13.5, Linux x86_64. Synthetic streaming demo, not live production traces.

| Metric | Observed |
|---|---:|
| Traces | 1,000,000 |
| Elapsed analysis | 4.798 s |
| Timeout loop | 50,000 |
| Invalid tool arguments | 50,000 |
| Truncated context | 50,000 |
| Wrong-order silent failure | 50,000 |
| Unverified | 800,000 |
| Model requests | 0 |
| API cost | $0 |
| Unit tests passed | 10/10 |
| Unit test time | 0.017 s |

A separate 1,000-trace demo replay matched all 1,000 IDs and removed 200 observable signals in synthetic counterfactual data. This does not establish real-world fix efficacy.

Reproduce:

```sh
python -m unittest -v
python reliability.py --n 1000000 --out benchmark.json --html benchmark.html
python reliability.py --n 1000 --replay-demo
```

The benchmark is deterministic rule-mode. The provider integration tests use mocked network responses; real Ollama/cloud inference was not executed in this environment. Wall time varies by hardware. No credentials were used.
