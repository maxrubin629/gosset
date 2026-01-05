from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
from collections import defaultdict


@dataclass
class TokenAgg:
    token: str
    count: int
    mean_entropy_bits: float
    mean_entropy_nats: float


def load_log(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_range(n_steps: int, start: Optional[int], end: Optional[int]) -> Tuple[int, int]:
    """Normalize a possibly-partial inclusive [start, end] range.

    Returns a clamped (start_i, end_i) pair such that:
      - 0 <= start_i <= end_i <= n_steps-1, when n_steps > 0
      - start_i == 0 and end_i == -1, when n_steps == 0
    """
    if n_steps <= 0:
        return 0, -1
    s = 0 if start is None else int(start)
    e = (n_steps - 1) if end is None else int(end)
    s = max(0, min(n_steps - 1, s))
    e = max(0, min(n_steps - 1, e))
    if e < s:
        s, e = e, s
    return s, e


def slice_steps(steps: List[Dict[str, Any]], start: Optional[int], end: Optional[int]) -> List[Dict[str, Any]]:
    if not steps:
        return []
    s, e = normalize_range(len(steps), start, end)
    if e < s:
        return []
    return steps[s : e + 1]


def token_stats(
    log: Dict[str, Any],
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    min_count: int = 5,
    top_n: int = 50,
) -> Dict[str, Any]:
    all_steps = log.get("steps", [])
    s_norm, e_norm = normalize_range(len(all_steps), start, end)
    steps = slice_steps(all_steps, start, end)
    by_tok = defaultdict(lambda: {"count": 0, "sum_bits": 0.0, "sum_nats": 0.0})

    for s in steps:
        t = s["token"]
        by_tok[t]["count"] += 1
        by_tok[t]["sum_bits"] += float(s.get("entropy_bits", 0.0))
        by_tok[t]["sum_nats"] += float(s.get("entropy_nats", 0.0))

    aggs: List[TokenAgg] = []
    for tok, v in by_tok.items():
        c = int(v["count"])
        if c < min_count:
            continue
        aggs.append(TokenAgg(
            token=tok,
            count=c,
            mean_entropy_bits=v["sum_bits"] / c,
            mean_entropy_nats=v["sum_nats"] / c,
        ))

    aggs.sort(key=lambda a: a.mean_entropy_bits, reverse=True)

    return {
        "range": {"start": s_norm, "end": e_norm, "count_steps": len(steps)},
        "min_count": min_count,
        "top": [asdict(a) for a in aggs[:top_n]],
    }


def summarize_range(log: Dict[str, Any], start: Optional[int], end: Optional[int]) -> Dict[str, Any]:
    all_steps = log.get("steps", [])
    s_norm, e_norm = normalize_range(len(all_steps), start, end)
    steps = slice_steps(all_steps, start, end)
    if not steps:
        return {"start": s_norm, "end": e_norm, "count_steps": 0}
    bits = [float(s.get("entropy_bits", 0.0)) for s in steps]
    nats = [float(s.get("entropy_nats", 0.0)) for s in steps]

    def pct(vals: List[float], p: float) -> float:
        vals_sorted = sorted(vals)
        if not vals_sorted:
            return 0.0
        k = (len(vals_sorted) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return vals_sorted[int(k)]
        return vals_sorted[f] * (c - k) + vals_sorted[c] * (k - f)

    return {
        "start": s_norm,
        "end": e_norm,
        "count_steps": len(steps),
        "entropy_bits": {
            "mean": sum(bits) / len(bits),
            "min": min(bits),
            "p50": pct(bits, 50),
            "p90": pct(bits, 90),
            "max": max(bits),
        },
        "entropy_nats": {
            "mean": sum(nats) / len(nats),
            "min": min(nats),
            "p50": pct(nats, 50),
            "p90": pct(nats, 90),
            "max": max(nats),
        },
    }
