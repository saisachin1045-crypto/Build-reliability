# Integrated reliability extension

The original archive is preserved in commit 7ca60103459761157595522650e2324d28a8c3c7. All 33 tracked files were verified against their original Git blob hashes. The extension plugs the provider adapter into the original pipeline, rather than replacing session joining, stratified sampling, statistical estimation and incident analysis.

## Run without keys (Python 3.10+)

    python -m unittest -v
    python -m trace_triage.live --n 1000

Open integrated-report.html to inspect incidents. This heuristic demo does not read synthetic truth, and unflagged observations are provisional passes, not proof of success. The original simulated evaluation is still available via python -m trace_triage demo.

## Optional real local AI

Install Ollama and pull qwen2.5:3b. Optionally pull nomic-embed-text for incident similarity. Start the local Ollama server, then:

    python -m trace_triage.live --provider ollama --n 1000 --max-calls 200 --embeddings

No API key is needed. Cloud alternatives are --provider openai and --provider claude with respective environment credentials OPENAI_API_KEY and ANTHROPIC_API_KEY. Never commit secrets. Model names are configurable with --model. Model call limits fail explicitly if exhausted during sampling; increase the cap if needed. Usage reports actual provider token counters when available; cost is estimated from configured unit prices. Live providers have not been tested against running servers in this environment.

## Recorded traces and verification

    python -m trace_triage.live --input traces.jsonl --after fixed.jsonl

JSONL fields follow trace_triage/models.py. Each span must have kind and name. Supply trace_id, user_goal, final_answer and spans; session_id, ts, and attrs have defaults. Ground truth is stripped from model input. Replay compares observed before/after signals, not live agent re-execution. Redaction is best effort, not a guarantee of anonymization.

## Scale and local checks

    python reliability.py --n 1000000 --out benchmark.json --html benchmark.html

The streaming benchmark is separate from the original in-memory sampling pipeline. See BENCHMARK.md and PROTOTYPE.md for measured synthetic demo details and limitations. Embedding similarity links incident representatives, rather than clustering all traces. The integrated local suite passed 176 tests (162 original, 10 streaming prototype, 4 adapter tests); the integrated demo processed 1,000 original-generator traces with 69 sampled verdicts and zero model calls. No production-scale or real-model accuracy claims are made.
