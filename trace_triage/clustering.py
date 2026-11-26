"""Stage 2: group traces so the unit of analysis is a failure mode, not a trace."""
from __future__ import annotations

from typing import Dict, FrozenSet, List, Mapping, Sequence, Tuple

from .fingerprint import error_signature, shape_key
from .models import Trace
from .tripwires import LOUD_FLAGS

ClusterKey = Tuple[str, str]


def cluster_key(trace: Trace, flags: FrozenSet[str]) -> ClusterKey:
    """(shape, signature). Signature is the error template, else the strongest loud flag, else '-'.

    Weak signals (user thumbs-down, token or latency outliers) never split a cluster:
    they would scatter one failure mode across many clusters.
    """
    signature = error_signature(trace)
    if signature is None:
        for flag in LOUD_FLAGS:
            if flag in flags:
                signature = f"flag:{flag}"
                break
    return shape_key(trace), signature or "-"


def cluster(traces: Sequence[Trace], flags: Mapping[str, FrozenSet[str]]) -> Dict[ClusterKey, List[Trace]]:
    groups: Dict[ClusterKey, List[Trace]] = {}
    for t in traces:
        groups.setdefault(cluster_key(t, flags[t.trace_id]), []).append(t)
    return groups
