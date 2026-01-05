from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests


TokenProb = Tuple[str, float]  # (token, logprob)


def sse_events(resp: requests.Response) -> Iterator[Dict]:
    """Yield JSON dicts from a text/event-stream ("data: {json}") response.

    NOTE: llama.cpp's SSE responses are UTF-8 JSON, but the server may omit an explicit
    `charset=utf-8`. If we let `requests` guess the encoding (via `decode_unicode=True`),
    it can default to latin-1 and produce mojibake (e.g. "🦉" -> "ð¦").

    We therefore decode lines explicitly as UTF-8.
    """
    for raw in resp.iter_lines(decode_unicode=False):
        if not raw:
            continue

        # `iter_lines(decode_unicode=False)` yields bytes. Decode explicitly to avoid
        # requests' charset detection.
        if isinstance(raw, (bytes, bytearray)):
            try:
                line = raw.decode("utf-8")
            except UnicodeDecodeError:
                # Be robust to any malformed bytes; better to keep going than to crash.
                line = raw.decode("utf-8", errors="replace")
        else:
            line = str(raw)

        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            yield json.loads(data)
        except json.JSONDecodeError:
            continue


def request_chat(
    base: str,
    model: str,
    prompt: str,
    system: Optional[str],
    kmax: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    timeout: float,
    stream: bool,
) -> Iterable[Dict]:
    """Yield server responses (streaming events or a single dict) from /v1/chat/completions."""
    url = f"{base}/v1/chat/completions"
    body = {
        "model": model,
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}],
        "stream": bool(stream),
        "logprobs": True,
        "top_logprobs": int(kmax),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
    }
    r = requests.post(url, json=body, stream=stream, timeout=timeout)
    r.raise_for_status()
    if stream:
        yield from sse_events(r)
    else:
        yield r.json()


def request_completion(
    base: str,
    prompt: str,
    kmax: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    max_tokens: int,
    timeout: float,
    stream: bool,
    post_sampling_probs: bool = False,
) -> Iterable[Dict]:
    """Yield server responses (streaming events or a single dict) from native /completion."""
    url = f"{base}/completion"
    body = {
        "prompt": prompt,
        "n_predict": int(max_tokens),
        "stream": bool(stream),
        "n_probs": int(kmax),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repeat_penalty": float(repeat_penalty),
        "post_sampling_probs": bool(post_sampling_probs),
    }
    r = requests.post(url, json=body, stream=stream, timeout=timeout)
    r.raise_for_status()
    if stream:
        yield from sse_events(r)
    else:
        yield r.json()


def extract_top_candidates_from_chat_event(evt: Dict) -> List[Tuple[str, List[TokenProb]]]:
    """Extract chosen token(s) and their top candidates from a chat-completions streaming chunk.

    llama.cpp's OpenAI-compatible streaming can return multiple token logprob entries per chunk.
    We return a list so callers can log 1 step per token.
    """
    choices = evt.get("choices") or []
    if not choices:
        return []
    ch0 = choices[0]
    lp = (ch0.get("logprobs") or {}).get("content", [])
    if not lp:
        return []

    outs: List[Tuple[str, List[TokenProb]]] = []
    for item in lp:
        chosen_tok = item.get("token")
        top = item.get("top_logprobs") or []
        topk: List[TokenProb] = []
        for cand in top:
            tok = cand.get("token", "")
            logprob = float(cand.get("logprob"))
            topk.append((tok, logprob))
        if chosen_tok is None or not topk:
            continue
        outs.append((str(chosen_tok), topk))
    return outs


def extract_top_candidates_from_completion_event(evt: Dict) -> List[Tuple[str, List[TokenProb]]]:
    """Extract chosen token and top candidates from /completion streaming chunk.

    In practice, /completion chunks usually contain a single new token, so we return a list with
    at most one item.
    """
    probs = evt.get("completion_probabilities")
    if not probs:
        return []
    last = probs[-1]
    chosen_tok = last.get("token") or ""
    top = last.get("top_logprobs")
    post_mode = False
    if top is None:
        top = last.get("top_probs") or []
        post_mode = True
    topk: List[TokenProb] = []
    for cand in top:
        tok = cand.get("token", "")
        if post_mode:
            p = float(cand.get("prob"))
            logprob = math.log(max(p, 1e-45))
        else:
            logprob = float(cand.get("logprob"))
        topk.append((tok, logprob))
    if not chosen_tok or not topk:
        return []
    return [(str(chosen_tok), topk)]


def entropy_lower_bound_from_topk(topk: List[TokenProb]) -> Tuple[float, float, float]:
    """Compute a lower bound on entropy given top-k probabilities.

    We only know probabilities for a subset S. Let m = sum_{i in S} p_i.
    The unknown tail has mass p_tail = 1 - m. If we merge the entire tail into one
    bucket, entropy decreases, so:

        H_true >= H_merged = -sum_{i in S} p_i log p_i - p_tail log p_tail

    Returns: (H_nats_lb, H_bits_lb, mass_observed)
    """
    ps = [math.exp(lp) for _, lp in topk]
    m = sum(ps)
    p_tail = max(0.0, 1.0 - m)

    h = 0.0
    for p in ps:
        if p > 0:
            h -= p * math.log(p)
    if p_tail > 0:
        h -= p_tail * math.log(p_tail)

    return h, h / math.log(2.0), m


@dataclass
class LlamaCppConfig:
    base_url: str
    model: str = ""
    endpoint: str = "chat"  # chat | completion
    kmax: int = 512
    timeout: float = 600.0
    max_tokens: int = 1024
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    repeat_penalty: float = 1.0
    stream: bool = True
    system: Optional[str] = None


def generate_with_entropy_lower_bound(
    *,
    cfg: LlamaCppConfig,
    prompt: str,
    out_path: Path,
) -> Dict[str, Any]:
    """Stream a completion from a llama.cpp server and log entropy lower bounds per token."""
    steps: List[Dict[str, Any]] = []
    text_out = []

    t0 = time.time()

    if cfg.endpoint == "chat":
        events = request_chat(
            base=cfg.base_url,
            model=cfg.model,
            prompt=prompt,
            system=cfg.system,
            kmax=cfg.kmax,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            stream=cfg.stream,
        )
        extractor = extract_top_candidates_from_chat_event
    else:
        events = request_completion(
            base=cfg.base_url,
            prompt=prompt,
            kmax=cfg.kmax,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            top_k=cfg.top_k,
            repeat_penalty=cfg.repeat_penalty,
            max_tokens=cfg.max_tokens,
            timeout=cfg.timeout,
            stream=cfg.stream,
            post_sampling_probs=False,
        )
        extractor = extract_top_candidates_from_completion_event

    idx = 0
    # Some llama.cpp servers return cumulative `logprobs.content` in each SSE chunk for the
    # OpenAI-compatible chat endpoint (i.e., all tokens so far, not just the new delta).
    # If we naively log every entry in every chunk, the output becomes duplicated/garbled.
    #
    # We keep a prefix of already-seen tokens and, when a chunk looks cumulative, we only
    # consume the new suffix.
    seen_chat_tokens: List[str] = []
    for evt in events:
        extracted = extractor(evt)
        if not extracted:
            continue

        if cfg.endpoint == "chat" and seen_chat_tokens:
            # Detect cumulative chunks: the chunk's token sequence starts with what we've
            # already logged.
            if len(extracted) >= len(seen_chat_tokens):
                is_prefix = True
                for i, prev_tok in enumerate(seen_chat_tokens):
                    if extracted[i][0] != prev_tok:
                        is_prefix = False
                        break
                if is_prefix:
                    extracted = extracted[len(seen_chat_tokens):]

        for tok, topk in extracted:
            h_nats_lb, h_bits_lb, mass_obs = entropy_lower_bound_from_topk(topk)
            steps.append({
                "index": idx,
                "token_id": None,
                "token": tok,
                "entropy_nats": h_nats_lb,
                "entropy_bits": h_bits_lb,
                "mass_observed": mass_obs,
                "kmax": cfg.kmax,
            })
            text_out.append(tok)
            if cfg.endpoint == "chat":
                seen_chat_tokens.append(tok)
            idx += 1

    dt = time.time() - t0

    log: Dict[str, Any] = {
        "schema_version": 1,
        "backend": "llamacpp_topk_lower_bound",
        "model_id": cfg.model or None,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "entropy is a LOWER BOUND when computed from top-k only",
        "prompt": prompt,
        "system": cfg.system,
        "decode": {
            "temperature": cfg.temperature,
            "top_k": cfg.top_k,
            "top_p": cfg.top_p,
            "min_p": 0.0,
            "repetition_penalty": cfg.repeat_penalty,
            "max_new_tokens": cfg.max_tokens,
            "seed": None,
        },
        "timing": {"seconds": dt, "tokens_per_second": (len(steps) / dt) if dt > 0 else None},
        "steps": steps,
        "response_text": "".join(text_out),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log
