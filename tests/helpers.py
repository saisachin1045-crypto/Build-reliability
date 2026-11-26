"""Small builders so tests read as scenarios, not constructor calls."""
from trace_triage.models import Span, Trace, Truth


def span(kind="llm", name="m", **kw):
    return Span(kind=kind, name=name, **kw)


def lookup_spans(args_hash="a1"):
    return [span("llm"), span("tool", "get_order", args_hash=args_hash), span("llm")]


def make_trace(
    trace_id="t1",
    *,
    spans=None,
    session_id="s1",
    ts=0,
    goal="hello",
    final="ok",
    user_signal=None,
    attrs=None,
    truth=None,
):
    return Trace(
        trace_id=trace_id,
        session_id=session_id,
        ts=ts,
        attrs=attrs or {},
        spans=tuple(spans if spans is not None else lookup_spans()),
        user_goal=goal,
        final_answer=final,
        user_signal=user_signal,
        truth=truth,
    )


def failing_truth(mode="tool_timeout_loop", first_bad_span=1):
    return Truth(failure_mode=mode, first_bad_span=first_bad_span)
