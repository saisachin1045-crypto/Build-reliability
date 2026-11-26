# agent-trace-triage

A small, runnable answer to one question:

> Say you have a million agent traces. How would you figure out what's failing and why, at the lowest possible cost?

**Short answer:** don't read a million traces. Build a funnel where every stage exists so the next, pricier stage reads less. Judge about 2% of the traces, and spend frontier-model money only on the few dozen clusters that matter. The full written answer is in [ANSWER.md](ANSWER.md). This repo is the part of it I could build and test without real traces.

## What this is, and what it is not

- **Is:** a dependency-free Python implementation (standard library only, about 1,450 lines plus about 1,240 lines of tests) of stages 0 to 4, an evaluation harness, and the cost model. It runs on synthetic traces with four seeded failure modes (three loud, one silent), so every stage can be scored against known truth.
- **Is not:** production code or evidence of scale. There are no real traces and it never calls a model.
- **The judge is simulated.** `SimulatedJudge` stands in for an LLM judge with configurable error rates. The numbers in `TIERS` are my assumptions, not measurements. The `Judge` protocol in `judge.py` is where a real judge plugs in, and `calibrate()` is how you would measure it.
- **Described but not built:** the frontier-model root-cause step (`Incident.brief()` builds its input), replaying a fix before and after (stage 5), and embedding-based clustering (clusters here use trace shape plus error signature).
- Every result below is on data I designed. It shows the mechanics work and where they break. It does not show how this performs on yours.

## The funnel

```
1,000,000 traces
  stage 0  join into sessions        a user's reaction often lands in the next trace
  stage 1  tripwires                 span metadata only, no model, no ground truth
  stage 2  cluster                   trace shape + error template: the unit becomes a failure mode
  stage 3  sample and judge          stratified sample per cluster -> weighted estimate with an interval
           + clean-pool audit        judge a random sample of the traces that tripped nothing
  stage 4  rank, then no-model RCA   slices, change point, failing-vs-passing contrast per incident
           -> brief                  a ~500-token package for a frontier model, top incidents only
  stage 5  verify by replay          described in ANSWER.md, not built
```

## Run it

```
python -m unittest discover -s tests -t .      # 162 tests, about 2 seconds
python -m trace_triage demo                    # 20,000 synthetic traces, scored against the seeded faults
python -m trace_triage demo --judge small      # same run with a weak judge (see findings)
python -m trace_triage sweep --seeds 20        # how often does it get every mode right?
python -m trace_triage cost                    # the cost model for a million traces
```

Python 3.10 or newer. No installs.

## The seeded failure modes

| mode | kind | cause (seeded) |
|---|---|---|
| `tool_timeout_loop` | loud | lookup agent, only after deploy `d42`: the tool times out and the agent retries in a loop |
| `bad_tool_args` | loud | booking agent on prompt `v37`: invalid tool arguments, then a wrong "confirmed" |
| `context_truncated` | loud | research agent with very long input: the answer is cut off (`finish=length`) |
| `wrong_customer_silent` | silent | lookup agent on prompt `v36`: answers about the wrong order, nothing errors |

Healthy traces include a transient error that retried fine, because an error is not a failure.

## What the demo prints

Real output of `python -m trace_triage demo` (seed 7):

```
Judge calibration on a golden set of 300 labelled traces (simulated judges: the error rates are assumptions, replace them with measured ones)
  judge          agreement  recall precision class acc
  small                74%     55%       89%       63%
  mid                  94%     91%       96%       82%
  frontier             98%     96%       99%       98%
  mid->frontier        99%     99%      100%      100%
  escalation rate of mid->frontier on the golden set: 14%; mid judge flips its pass/fail verdict between two runs on 10% of traces
  cheapest judge clearing recall>=90%, precision>=90%, class accuracy>=80%: mid->frontier

Traces 20,000 | tripped a wire 1,955 (9.8%) | clusters 8 | judged 1,484 (7.4%)
Judge mid->frontier: 1,484 calls, 65 escalated one tier up
Clean-pool audit: 1,000 of 18,045 untripped traces judged, 6 failed -> 0.6% [0.3%, 1.3%], about 108 silent failures

Incidents
#1 tool_timeout_loop   cluster: llm>tool:get_order*>llm | timeout after <n>ms calling orders-api
    size 909, failure rate 100.0% [87.2%, 100.0%], about 909 failures (10 judged)
    top slice: deploy=d42 (36% vs 0% elsewhere, explains ~909; exact labels on 6,192 traces)
    changepoint: ts 12,004 (0% -> 36%)
    contrast: span 1: failing tool:get_order timeout vs passing tool:get_order ok
#2 wrong_customer_silent   cluster: llm>tool:get_order>llm | -
    size 5,188, failure rate 7.9% [2.3%, 13.6%], about 412 failures (75 judged)
    top slice: prompt_version=v36 (19% vs 2% elsewhere, explains ~361; 145 judged samples, weighted)
    contrast: no structural divergence: the spans match, so the failure is in content and needs a content-level review
#3 bad_tool_args   cluster: llm>tool:check_slots>llm>tool:book>llm | <n> invalid slot_id: <n>
    size 315, failure rate 100.0% [87.3%, 100.0%], about 315 failures (10 judged)
    top slice: prompt_version=v37 (29% vs 0% elsewhere, explains ~315; exact labels on 3,081 traces)
    contrast: span 3: failing tool:book error args=invalid vs passing tool:book ok
#4 context_truncated   cluster: llm>retrieval:kb>llm | flag:truncated
    size 220, failure rate 100.0% [87.4%, 100.0%], about 220 failures (10 judged)
    top slice: input_len=xl (59% vs 0% elsewhere, explains ~220; exact labels on 3,918 traces)
    contrast: span 2: failing llm:m ok finish=length vs passing llm:m ok

Scored against the seeded failure modes (ground truth, never read by stages 1, 2 and 4)
  mode                     failing  any wire  metadata  surfaced     est  root cause
  tool_timeout_loop            909      100%      100%       yes     909         yes
  bad_tool_args                315      100%      100%       yes     315         yes
  context_truncated            220      100%      100%       yes     220         yes
  wrong_customer_silent        364       43%        0%       yes     412         yes
  failures that tripped no wire: 206 (the audit estimated 108)
```

## Findings from building and running it

1. **It works on this data, repeatedly.** Over 20 seeds of 20,000 traces, all four modes were surfaced as incidents and every top slice was the seeded driver. An earlier 10-seed sweep got the silent mode's root cause right in 9 of 10. In the miss, every sampled failure happened to share both a deploy and a prompt version, so two slices explained the same failures exactly. The tie-break now prefers the narrower slice. Real data will confound more often than this generator does, and a replay is what separates confounded causes.
2. **Cost follows clusters, not traces.** The number of judged traces barely moves as the input grows: 1,360 at 5,000 traces, 1,484 at 20,000 and 1,535 at 100,000. That is 27%, then 7.4%, then 1.5% of the traces.
3. **Silent failures are the hard part.** No silent failure tripped a metadata wire, and 43% tripped a user-signal wire. 206 of 364 tripped nothing. The audit's point estimate of those (108) was about half the truth, but its interval, 54 to 235, contained the 206. Trust the interval. With few hits the point estimate is noisy.
4. **A weak judge makes phantom incidents.** With the small tier only (`demo --judge small`; assumed 55% recall, 5% false positives, 60% class accuracy), the same run produced three phantom incidents in healthy clusters. It estimated 830 silent failures against a true 206, undercounted the loud modes (500 timeouts against a true 909), and got two root causes wrong. This is a simulation of my assumptions, so read it as how judge errors propagate, not as a measurement of any model.
5. **Bugs the end-to-end tests caught while I built it:**
   - The simulated judge tiers shared one random stream, so their mistakes were perfectly correlated and escalating a low-confidence verdict fixed nothing.
   - Judge false negatives sat in the passing pool and became the "closest passing trace", hiding the real divergence. Contrast now uses only confident passes.
   - Using the loop-collapsed shape as the baseline for a change point left no "before" period. The baseline is now every trace doing the same job (same tools).
   - Change points from about ten sampled failures were off by thousands of minutes. Loud clusters are labelled for free by their tripwire, so they use exact labels and nothing is sampled.
   - A judge whose measured recall merely sat near the bar passed on a 300-trace golden set. The bar is now checked against the 95% lower bound, so a small golden set cannot certify a judge.

## Cost model

`python -m trace_triage cost` prints this. Every figure is an assumption you can change in `trace_triage/costs.py`. Prices are vendor list prices as cached on 2026-11-03, in USD per 1M tokens, so check them before relying on them. Batch is 50% off every token and stacks with caching. Thinking tokens bill as output, so the stage 3 figure is a lower bound if the judge runs with thinking on.

```
Cost model for 1,000,000 traces (60,000 raw tokens each, 22,000 judged). Assumptions, not measurements.
  1 tripwires (compute)          $       20 - $       20
  2 embeddings for clustering    $       10 - $       65
  3 sampled judging (batch)      $      121 - $      242
  4 root cause, top incidents    $      100 - $      300
  5 verify fixes by replay       $      100 - $      200
  TOTAL                          $      351 - $      827
  at most $0.00083 per trace

For comparison, judging every raw trace instead (input tokens only):
  haiku-5.5    $     6,000   on batch $     3,000
  sonnet-5.5   $   120,000   on batch $    60,000
  opus-5.5     $   240,000   on batch $   120,000
```

The funnel is not there to save pennies on a small model. A small model over every trace is the cheapest line above, but its misses are invisible to you. The funnel makes the best model affordable on the clusters where it matters.

## Layout

| module | role |
|---|---|
| `models.py` | `Span`, `Trace`, `Truth` (ground truth, synthetic only) |
| `synth.py` | synthetic traces with the four seeded failure modes |
| `sessions.py` | stage 0: mark a trace `rephrase` when the next one in its session repeats the goal |
| `tripwires.py` | stage 1: error, timeout, retry, loop, invalid args, truncation, outliers, user signals |
| `fingerprint.py`, `clustering.py` | stage 2: shape key, error template, workflow key, cluster key |
| `sampling.py`, `stats.py` | stage 3: stratified sampling with weights, widening rule, Wilson intervals, clean-pool audit |
| `judge.py` | stage 3: `Judge` protocol, simulated judge, tier escalation, calibration, flakiness |
| `compress.py` | stage 3: loops collapsed, PII redacted, long text cut; ground truth never included |
| `rootcause.py` | stage 4: slices, change point, contrast pair (no model) |
| `pipeline.py` | the whole run, incident briefs, scoring against the seeded faults |
| `costs.py` | the cost model |

## Limits

- Synthetic data and a simulated judge. Nothing here has seen a real trace or a real model.
- Clusters use shape plus error signature only. A silent failure inside a healthy-looking cluster is found by sampling, not by clustering, and several silent modes sharing one cluster would blur together.
- One change point per incident, and only increases. Slices can be confounded when few failures were sampled.
- PII handling is two regexes. Do not treat it as a redaction guarantee.
- Stage 5, the frontier root-cause agent and embedding clustering are described in [ANSWER.md](ANSWER.md), not implemented.
