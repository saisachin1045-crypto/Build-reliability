"""Small statistics helpers shared by sampling and judge calibration."""
from __future__ import annotations

import math
from typing import Tuple


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95% interval for a proportion k/n. Honest at 0/n and n/n, unlike the normal approximation."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)
