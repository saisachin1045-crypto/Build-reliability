# Agent Trace Reliability

A runnable, standard-library Python prototype for cost-aware triage of AI agent traces. No account, API key, pip install or network is required for demo mode. This is a standalone implementation of the trace triage concept, not a claim of production readiness.

## Quick start

Python 3.10 or newer:

```sh
python reliability.py --n 1000 --replay-demo
python -m unittest -v
```

Open report.html in a browser to filter and expand incidents, view deployment rates, evidence, hypotheses, model usage and replay results. The JSON output is report.json.

## Real local AI (no API keys)

Install Ollama from https://ollama.com and start its local service. Download a model once (network access and disk space required):

```sh
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
python reliability.py --provider ollama --model qwen2.5:3b --embeddings --n 1000 --max-calls 24
```

The model selects an allowlisted read-only evidence tool (trace inspection, deployment comparison or counts), receives its result and proposes a diagnosis. It also audits a reservoir sample of unflagged traces. Embedding mode computes cosine similarity links between incident representatives; it does not merge incompatible failure classes automatically. No shell or write tools are exposed. Model errors are recorded and deterministic analysis remains available.

## Optional cloud providers

Use --provider openai --model YOUR_MODEL or --provider claude --model YOUR_MODEL and set OPENAI_API_KEY or ANTHROPIC_API_KEY in the environment. No credentials are checked in. These modes send redacted evidence externally; use only data approved for that purpose. Set INPUT_USD_PER_MILLION and OUTPUT_USD_PER_MILLION to your current provider rates to calculate an estimated cost from reported token counts. Model call limits and in-memory caching bound the number of requests. Local compute costs are not included.

## Analyze your own traces

Supply newline-delimited JSON, one object per line:

```json
{"trace_id":"case-1","user_goal":"Where is order 1234?","final_answer":"Order 1235 is shipped.","spans":[{"kind":"tool","name":"lookup","status":"ok"}],"attrs":{"deploy":"v2"}}
```

```sh
python reliability.py --input traces.jsonl
python reliability.py --input before.jsonl --after after.jsonl
```

Trace IDs should be unique per run and stable across replay runs. The optional after file must contain observations from rerunning the same cases on a modified agent. Replay compares observable failure signals across matched IDs. It does not automatically rerun arbitrary external agents and absence of a signal does not prove a successful outcome. The demo replay is a synthetic counterfactual.

## One million trace benchmark

```sh
python reliability.py --n 1000000 --out benchmark.json --html benchmark.html
```

The seeded-pattern generator streams one million synthetic records without retaining them all. Expected distribution: 800,000 unverified and 50,000 each of timeout loops, invalid tool arguments, truncated context and wrong-order responses. The program measures elapsed processing time on your hardware; CI runs the same benchmark and asserts the counts. This is not a benchmark of production traffic or one million model calls.

## Method and limitations

1. Observable signals are checked without using synthetic ground truth. Successful-looking traces are **unverified**, not automatically marked as passing.
2. Failure classes group recurring signals and preserve an example; deployment rates support causal hypotheses, not proof.
3. Optional Ollama embeddings add semantic similarity links. The bounded model audit can surface silent failures not covered by the rules.
4. Replay checks observed before/after signals and missing cases.
5. PII masking covers common email, phone, IP, SSN and secret patterns, not every possible identifier or sensitive field. Do not treat it as comprehensive de-identification.

Provider adapter tests mock network responses; actual local/cloud model quality, latency and token costs need to be measured with a running model and labeled real traces. The synthetic fixture has deliberately simple failure patterns; it is not evidence of general real-world detection accuracy. Dashboard output can contain sensitive business evidence even after masking; do not publish real reports without review.
