"""Plain data model for one agent run: a trace made of spans."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class Span:
    """One step: a model call, a tool call or a retrieval."""

    kind: str  # "llm" | "tool" | "retrieval"
    name: str
    status: str = "ok"  # "ok" | "error" | "timeout"
    error: Optional[str] = None
    latency_ms: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    args_hash: str = ""  # identifies the tool arguments, so repeats can be spotted
    finish_reason: str = "stop"  # "stop" | "length"
    args_valid: bool = True  # tool arguments passed schema validation
    result_count: Optional[int] = None  # retrieval hits


@dataclass(frozen=True)
class Truth:
    """Ground truth, present only on synthetic traces.

    Only the simulated judge and the evaluator read it. Tripwires, clustering,
    sampling and root-cause slicing never do.
    """

    failure_mode: Optional[str]
    first_bad_span: Optional[int] = None


@dataclass(frozen=True)
class Trace:
    trace_id: str
    session_id: str
    ts: int  # minute index; only ordering matters
    attrs: Mapping[str, str]  # prompt_version, model, deploy, tier, locale, input_len
    spans: Tuple[Span, ...]
    user_goal: str
    final_answer: Optional[str]
    user_signal: Optional[str] = None  # thumbs_down | rephrase | escalated | abandoned
    truth: Optional[Truth] = field(default=None, compare=True)
