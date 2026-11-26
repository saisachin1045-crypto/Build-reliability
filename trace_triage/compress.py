"""Compress a trace into the few lines a judge actually needs: loops collapsed, PII redacted, long text cut.

Ground truth is never included. Span indexes are kept so a verdict can cite 'the first bad span'.
"""
from __future__ import annotations

import re

from .fingerprint import normalise_error
from .models import Trace

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")


def redact(text: str) -> str:
    text = _EMAIL.sub("<email>", text)
    return _PHONE.sub(lambda m: "<phone>" if sum(c.isdigit() for c in m.group()) >= 9 else m.group(), text)


def head_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]} [...{len(text) - 2 * half} chars...] {text[-half:]}"


def estimate_tokens(text: str) -> int:
    """Rough rule of thumb: four characters per token."""
    return len(text) // 4


def _same(a, b) -> bool:
    return (a.kind, a.name, a.status, a.args_hash, a.error, a.finish_reason, a.args_valid) == (
        b.kind, b.name, b.status, b.args_hash, b.error, b.finish_reason, b.args_valid,
    )


def compress(trace: Trace) -> str:
    lines = [f"goal: {head_tail(redact(trace.user_goal), 200)}"]
    spans = trace.spans
    i = 0
    while i < len(spans):
        s = spans[i]
        j = i
        while j + 1 < len(spans) and _same(s, spans[j + 1]):
            j += 1
        repeats = j - i + 1
        line = f"[{i}] {s.kind}:{s.name} {s.status}"
        if repeats > 1:
            line += f" x{repeats}"
        if s.error:
            line += f' err="{normalise_error(s.error)}"'
        if s.finish_reason != "stop":
            line += f" finish={s.finish_reason}"
        if not s.args_valid:
            line += " args=invalid"
        if s.kind == "llm" and repeats == 1:
            line += f" tok={s.tokens_in}/{s.tokens_out}"
        lines.append(line)
        i = j + 1
    final = trace.final_answer
    lines.append("final: <none>" if final is None else f"final: {head_tail(redact(final), 200)}")
    if trace.user_signal:
        lines.append(f"signal: {trace.user_signal}")
    return "\n".join(lines)
