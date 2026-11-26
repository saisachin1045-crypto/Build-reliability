"""Stage 2 fingerprints: what a trace looked like (shape) and how it broke (error signature)."""
from __future__ import annotations

import re
from typing import Optional

from .models import Trace

_UUID = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_HEX = re.compile(r"\b[0-9a-f]{12,}\b")
_QUOTED = re.compile(r"(['\"]).*?\1")
_NUMBER = re.compile(r"(?<![a-z_])\d+(?:\.\d+)?")
LOOP = "*"  # ASCII on purpose: it is printed in reports and Windows consoles mangle anything else


def normalise_error(message: str) -> str:
    """Turn an error message into a template: ids, quoted strings and numbers become placeholders."""
    text = _UUID.sub("<id>", message.lower())
    text = _QUOTED.sub("<s>", text)
    text = _HEX.sub("<id>", text)
    text = _NUMBER.sub("<n>", text)
    return re.sub(r"\s+", " ", text).strip()


def shape_key(trace: Trace) -> str:
    """Ordered span kinds and tool names; model names ignored; consecutive repeats collapse to a loop mark."""
    tokens = []
    for s in trace.spans:
        token = "llm" if s.kind == "llm" else f"{s.kind}:{s.name}"
        if tokens and tokens[-1] in (token, token + LOOP):
            tokens[-1] = token + LOOP
        else:
            tokens.append(token)
    return ">".join(tokens)


def workflow_key(trace: Trace) -> str:
    """The job a trace does: the set of tools and retrievals it used, ignoring order and repeats.

    A loop and a single call share a workflow, so a failure's baseline is every trace doing the
    same job, not just the traces that happen to share its collapsed shape.
    """
    used = sorted({f"{s.kind}:{s.name}" for s in trace.spans if s.kind != "llm"})
    return ",".join(used) or "llm-only"


def error_signature(trace: Trace) -> Optional[str]:
    """Normalised message of the first failing span that carries one, else None."""
    for s in trace.spans:
        if s.status != "ok" and s.error:
            return normalise_error(s.error)
    return None
