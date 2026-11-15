"""
Small statistics helpers for reporting (stdlib only).

- wilson_ci:            95% interval for a proportion (ASR, ORR) in one slice
- paired_bootstrap_diff: difference between two checkpoints evaluated on the
                         SAME prompts (e.g. base vs repaired), with a CI
- cumulative_asr_by_turn: ASR@t for multi-turn replays (success by turn t)
"""

import math
import random
from typing import Dict, List, Optional, Sequence, Tuple


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_bootstrap_diff(a: Sequence[bool], b: Sequence[bool], iters: int = 2000,
                          seed: int = 0) -> Dict[str, float]:
    """
    Mean(b) - mean(a) over paired items with a percentile bootstrap 95% CI.
    a and b must be aligned (same prompt at the same index).
    """
    if len(a) != len(b):
        raise ValueError("paired_bootstrap_diff needs aligned sequences")
    n = len(a)
    if n == 0:
        return {"diff": 0.0, "lo": 0.0, "hi": 0.0, "n": 0}
    rng = random.Random(seed)
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] for i in idx) / n - sum(a[i] for i in idx) / n)
    diffs.sort()
    lo = diffs[int(0.025 * iters)]
    hi = diffs[min(iters - 1, int(0.975 * iters))]
    return {"diff": sum(b) / n - sum(a) / n, "lo": lo, "hi": hi, "n": n}


def cumulative_asr_by_turn(per_turn_violation: List[List[bool]], max_turns: Optional[int] = None) -> List[float]:
    """
    per_turn_violation[i] = judge violation flags for conversation i, turn by turn.
    Returns [ASR@1, ASR@2, ...]: share of conversations with a violation at or before turn t.
    Conversations shorter than t keep their final status.
    """
    if not per_turn_violation:
        return []
    T = max_turns or max(len(v) for v in per_turn_violation)
    out = []
    for t in range(1, T + 1):
        hits = sum(1 for v in per_turn_violation if any(v[:t]))
        out.append(hits / len(per_turn_violation))
    return out
