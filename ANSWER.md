# The answer

> Say you have a million agent traces. How would you figure out what's failing and why, at the lowest possible cost? (The question's hint: local and small models won't get you there.)

I'd optimise cost per confirmed fix, not cost per trace, and I wouldn't read a million traces. I'd read about 22,000, chosen to stand in for the rest. Each stage below exists so the next, pricier one reads less. The code in this repo implements stages 0 to 4 on synthetic traces; see the [README](README.md) for what is built and what is only described.

**Assumptions** (change them in `trace_triage/costs.py`): about 60k raw tokens per trace because agents resend their history on every call, about 15k after deduping repeated prompts and tool schemas, and about 4k after compressing for a judge. Prices are vendor list prices cached on 2026-11-03, per 1M input tokens: Haiku 5.5 $0.10, Sonnet 5.5 $2, Opus 5.5 $4. Batch is 50% off and stacks with caching.

## Stage 1: tripwires, no model (about $20 of compute)

Join traces into sessions first, since the user's reaction often lands in the next trace. Then, from span metadata alone: exceptions, tool errors and timeouts, retries, invalid tool arguments, truncated outputs, loops (same tool and arguments three or more times), step, token, cost and latency outliers against traces of the same shape, no final answer, and user signals (thumbs-down, rephrase, escalation, abandonment).

This says where to look, not why, and it misses silent failures, so stage 3 also samples the clean traces.

## Stage 2: cluster, so the unit is a failure mode, not a trace (about $10-65)

Fingerprint three ways: trace shape (ordered span kinds, tool names and statuses, with loops collapsed), error signature (exception type plus message template with ids stripped), and meaning (embed the first user message and the final answer, then k-means or HDBSCAN).

I'd expect a few thousand clusters because failures are heavy-tailed, but I'd check the cluster-size histogram before trusting that. Embeddings are the one place a small or local model is fine: the job is grouping, not judging.

## Stage 3: sample and judge (about $120 on Sonnet 5.5, about $240 on Opus 5.5, on batch)

Ten traces per cluster, mixing tripwire-positive, tripwire-negative, newest and furthest-from-centroid, widening to about 50 where a cluster is big or its failure rate is ambiguous. Add about 2,000 random clean traces, the only way to measure what the tripwires miss. With about 2,000 clusters that is about 22k judgments instead of a million.

Compress each trace first (collapse loops, keep the head and tail of big tool outputs, redact PII) so the judge reads about 4k tokens. The judge scores whether the user's goal was met, not whether anything errored. It picks a class from a fixed taxonomy and cites the first bad span, as structured output with the rubric as a cached prefix. Out comes a failure rate with an interval per cluster, scaled by cluster size.

The judge is mid-tier at minimum, and hard or low-confidence verdicts go to a frontier model. A small model gets the job only if it clears the golden-set bar below on goal-level verdicts, and given the hint I don't expect it to.

## Stage 4: root cause on the top 40 or so only (about $100-300)

Rank by failure rate, volume, severity and newness. Then, in order:

1. Slice the failure rate by prompt version, model, tool version, deploy, customer, locale, input length and hour, plus change points. Often the answer is right there, for free.
2. Contrast failing against passing traces of the same job and find the first step where they diverge.
3. Only then a frontier model with tools to query the trace store, asked for a cause, evidence spans, confidence and what would falsify it.

This is where the hint bites. The model has to hold the agent's prompt, its tool schemas and a 30-step trajectory, and find the causal step, not the loudest one. I'd expect small or local models to give a plausible wrong answer here, and I'd confirm that on a golden set rather than assume it. Put a hard cap per incident, about $5. Tool loops can't use batch, so this stage pays list price, but it is only 40 incidents.

## Stage 5: verify, then make it permanent (about $100-200, assumed)

A cause is only a hypothesis until a replay flips the outcome. Replay recorded inputs, with tool results stubbed from the trace, against the candidate fix. Judge before and after on the failing set plus a guard set of passing traces, then hand a fix brief to a human.

Each confirmed mode becomes a standing detector: a rule where possible, otherwise a small classifier trained on the frontier-labelled examples. Small models belong here and only here: screening for modes a frontier model already found and labelled, never discovering new ones, so the next million only send novel clusters up the funnel.

## Total

Roughly $350-850, under a tenth of a cent per trace. Judging every raw trace instead is about 60B tokens: about $120k on Sonnet 5.5 and about $240k on Opus 5.5, half that on batch. A small model over everything would be about $3-6k, but that is the shortcut the hint rules out: you'd inherit its misses with no way to see them. So the funnel isn't there to save pennies. It makes the best model affordable on the 40 clusters where it matters.

## What I'd skip

- LLM-judging healthy traces (audit them with the 2k sample instead).
- Per-trace explanations. Nobody acts on 40,000 of them.
- Fine-tuning or hosting a local judge before the labels are stable.
- A managed vector database. One million 384-float vectors is about 1.5 GB, so FAISS in memory is enough.
- Re-judging unchanged work. Cache on skeleton hash, rubric version and judge model.
- Anything real-time. It's a batch job on spot instances.

## How I'd know it works

- **A golden set** of about 300 hand-labelled traces (failure class and first bad span), to measure each tier's agreement and failure recall and pick the cheapest that clears the bar. Compare the 95% lower bound to the bar, not the measured value, so a small golden set can't certify a judge. Re-run it on every rubric or model change, and run the judge twice to measure its own flakiness.
- **Seeded faults.** Replay a few hundred traces with a planted bug (a broken tool, truncated context, a bad prompt version), mix them into the million, and see which stage loses each one. That is mutation testing for the pipeline, and it is what `python -m trace_triage demo` does on synthetic data.
- **Run metrics.** Cost per confirmed incident, incident precision (the share a human calls real and actionable), time to detect, and what the observer itself costs per trace. If you run dozens of these agents, their spend should be a first-class metric.
