"""Stage 0: join traces into sessions, because the user's reaction often lands in the next trace."""
from __future__ import annotations

import dataclasses
import re
from typing import Iterable, List

from .models import Trace


def _normalise(goal: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", goal.lower())).strip()


def join_sessions(traces: Iterable[Trace]) -> List[Trace]:
    """Mark a trace 'rephrase' when the next trace in its session repeats the same goal.

    Expects time-ordered input. Returns new traces; never overwrites an existing signal.
    """
    out = list(traces)
    last_index = {}
    for i, trace in enumerate(out):
        j = last_index.get(trace.session_id)
        if j is not None:
            earlier = out[j]
            if earlier.user_signal is None and _normalise(earlier.user_goal) == _normalise(
                trace.user_goal
            ):
                out[j] = dataclasses.replace(earlier, user_signal="rephrase")
        last_index[trace.session_id] = i
    return out
