"""Synthetic traces with seeded failure modes, so every stage can be scored against known truth.

Four modes, each tied to the attribute that causes it:

  tool_timeout_loop      loud    lookup agent, only after deploy d42: tool times out, agent retries in a loop
  bad_tool_args          loud    booking agent on prompt v37: invalid tool arguments, then a wrong "confirmed"
  context_truncated      loud    research agent with very long input: the answer is cut off (finish=length)
  wrong_customer_silent  silent  lookup agent on prompt v36: answers about the wrong order, nothing errors

Healthy traces include a transient error that retried fine (an error is not a failure).
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from .models import Span, Trace, Truth

FAILURE_MODES = ("tool_timeout_loop", "bad_tool_args", "context_truncated", "wrong_customer_silent")

DRIVERS = {
    "tool_timeout_loop": ("deploy", "d42"),
    "bad_tool_args": ("prompt_version", "v37"),
    "context_truncated": ("input_len", "xl"),
    "wrong_customer_silent": ("prompt_version", "v36"),
}

SHAPE_NAMES = ("faq", "lookup", "research", "booking")
SHAPE_WEIGHTS = (0.35, 0.30, 0.20, 0.15)
TOPICS = ("remote work", "travel", "expenses", "leave", "security", "training", "equipment", "benefits")
INPUT_LEN = ("s", "m", "l", "xl")
INPUT_LEN_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
INPUT_SCALE = {"s": 1.0, "m": 1.3, "l": 2.0, "xl": 4.0}


def _llm(rng: random.Random, scale: float = 1.0, finish: str = "stop", out: Optional[int] = None) -> Span:
    return Span(
        "llm",
        "m",
        latency_ms=int(rng.lognormvariate(6.8, 0.3)),
        tokens_in=int(rng.lognormvariate(7.5, 0.3) * scale),
        tokens_out=out if out is not None else int(rng.lognormvariate(5.0, 0.4)),
        finish_reason=finish,
    )


def _tool(rng, name, args_hash, status="ok", error=None, valid=True, latency=None) -> Span:
    return Span(
        "tool",
        name,
        status=status,
        error=error,
        latency_ms=latency if latency is not None else int(rng.lognormvariate(5.5, 0.4)),
        args_hash=args_hash,
        args_valid=valid,
    )


def _retrieval(rng) -> Span:
    return Span("retrieval", "kb", latency_ms=int(rng.lognormvariate(5.0, 0.4)), result_count=rng.randint(1, 8))


def _context(shape: str, rng: random.Random) -> Dict:
    if shape == "lookup":
        oid = rng.randint(1000, 9999)
        return {"goal": f"Where is my order {oid}?", "oid": oid}
    if shape == "booking":
        slot, cust = rng.randint(1, 200), rng.randint(1000, 9999)
        return {"goal": f"Book slot {slot} for customer {cust}", "slot": slot, "cust": cust}
    if shape == "research":
        topic = rng.choice(TOPICS)
        return {"goal": f"What is the policy on {topic} for region {rng.randint(1, 99)}?", "topic": topic}
    return {"goal": f"Question {rng.randint(1000, 9999)}: how do I reset my password?"}


def _make(shape: str, attrs: Dict[str, str], ctx: Dict, rng: random.Random) -> Tuple[List[Span], Optional[str], Optional[str], Optional[int]]:
    """Return spans, final answer, failure mode and the first bad span index."""
    if shape == "lookup":
        oid = ctx["oid"]
        h = f"oid{oid}"
        ok_answer = f"Your order {oid} is out for delivery."
        if attrs["deploy"] == "d42" and rng.random() < 0.35:
            loops = rng.randint(3, 5)
            spans = (
                [_llm(rng)]
                + [_tool(rng, "get_order", h, "timeout", "timeout after 30000ms calling orders-api", latency=30000) for _ in range(loops)]
                + [_llm(rng)]
            )
            final = None if rng.random() < 0.7 else "Sorry, I could not retrieve your order."
            return spans, final, "tool_timeout_loop", 1
        if attrs["prompt_version"] == "v36" and rng.random() < 0.20:
            other = oid + rng.randint(1, 50)
            spans = [_llm(rng), _tool(rng, "get_order", f"oid{other}"), _llm(rng)]
            return spans, f"Your order {other} is out for delivery.", "wrong_customer_silent", 1
        if rng.random() < 0.02:
            spans = [_llm(rng), _tool(rng, "get_order", h, "error", "503 service unavailable"), _tool(rng, "get_order", h), _llm(rng)]
            return spans, ok_answer, None, None
        return [_llm(rng), _tool(rng, "get_order", h), _llm(rng)], ok_answer, None, None

    if shape == "booking":
        slot, cust = ctx["slot"], ctx["cust"]
        if attrs["prompt_version"] == "v37" and rng.random() < 0.30:
            bad = rng.randint(1000, 9999)
            spans = [
                _llm(rng),
                _tool(rng, "check_slots", f"c{cust}"),
                _llm(rng),
                _tool(rng, "book", f"bad{bad}", "error", f"422 invalid slot_id: {bad}", valid=False),
                _llm(rng),
            ]
            return spans, "Your booking is confirmed.", "bad_tool_args", 3
        spans = [_llm(rng), _tool(rng, "check_slots", f"c{cust}"), _llm(rng), _tool(rng, "book", f"s{slot}"), _llm(rng)]
        return spans, f"Booked slot {slot}.", None, None

    if shape == "research":
        topic = ctx["topic"]
        scale = INPUT_SCALE[attrs["input_len"]]
        if attrs["input_len"] == "xl" and rng.random() < 0.60:
            spans = [_llm(rng, scale), _retrieval(rng), _llm(rng, scale, finish="length", out=4096)]
            return spans, f"The policy on {topic} states that employees must", "context_truncated", 2
        spans = [_llm(rng, scale), _retrieval(rng), _llm(rng, scale)]
        return spans, f"The policy on {topic} allows two days of remote work.", None, None

    return [_llm(rng)], "Open Settings, then Account, then Reset password.", None, None


def generate(n: int, seed: int = 7) -> List[Trace]:
    """Generate n time-ordered traces. The same seed always gives the same traces."""
    rng = random.Random(seed)
    traces: List[Trace] = []
    session = 0
    prev_failed = False
    prev_shape = "faq"
    prev_ctx: Dict = {}
    for i in range(n):
        new_session = i == 0 or rng.random() < 0.5
        if new_session:
            session += 1
        retry = (not new_session) and prev_failed and rng.random() < 0.5
        if retry:
            shape, ctx = prev_shape, prev_ctx
        else:
            shape = rng.choices(SHAPE_NAMES, SHAPE_WEIGHTS)[0]
            ctx = _context(shape, rng)
        attrs = {
            "prompt_version": rng.choices(("v35", "v36", "v37"), (0.30, 0.35, 0.35))[0],
            "model": rng.choice(("m-large", "m-mid")),
            "deploy": "d42" if i >= 0.6 * n else "d41",
            "tier": rng.choice(("free", "pro", "enterprise")),
            "locale": rng.choice(("en", "hi")),
            "input_len": rng.choices(INPUT_LEN, INPUT_LEN_WEIGHTS)[0],
        }
        spans, final, mode, first_bad = _make(shape, attrs, ctx, rng)
        failing = mode is not None
        signal = "thumbs_down" if rng.random() < (0.25 if failing else 0.01) else None
        traces.append(
            Trace(
                trace_id=f"t{seed}-{i:06d}",
                session_id=f"s{session}",
                ts=i,
                attrs=attrs,
                spans=tuple(spans),
                user_goal=ctx["goal"],
                final_answer=final,
                user_signal=signal,
                truth=Truth(mode, first_bad),
            )
        )
        prev_failed, prev_shape, prev_ctx = failing, shape, ctx
    return traces
