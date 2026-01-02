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


def slice_steps(steps: List[Dict[str, Any]], start: Optional[int], end: Optional[int]) -> List[Dict[str, Any]]:
    if start is None and end is None:
        return steps
    if start is None:
        start = 0
    if end is None:
        end = len(steps) - 1
    start = max(0, start)
    end = min(len(steps) - 1, end)
    if end < start:
        start, end = end, start
    return steps[start : end + 1]


def token_stats(
    log: Dict[str, Any],
    *,
    start: Optional[int] = None,
    end: Optional[int] = None,
    min_count: int = 5,
    top_n: int = 50,
) -> Dict[str, Any]:
    steps = slice_steps(log.get("steps", []), start, end)
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
        "range": {"start": start, "end": end, "count_steps": len(steps)},
        "min_count": min_count,
        "top": [asdict(a) for a in aggs[:top_n]],
    }


def summarize_range(log: Dict[str, Any], start: Optional[int], end: Optional[int]) -> Dict[str, Any]:
    steps = slice_steps(log.get("steps", []), start, end)
    if not steps:
        return {"count_steps": 0}
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
