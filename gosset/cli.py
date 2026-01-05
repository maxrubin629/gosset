from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional


# NOTE: Backends are imported lazily inside subcommands so that lightweight
# operations like `gosset analyze` don't require installing heavy deps
# (torch/transformers) or requests.


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    # stdin
    if sys.stdin.isatty():
        raise SystemExit("No prompt provided. Use --prompt, --prompt-file, or pipe via stdin.")
    return sys.stdin.read()


def cmd_generate(args: argparse.Namespace) -> None:
    try:
        from .backends.transformers_backend import DecodeConfig, generate_with_entropy
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependencies for the Transformers backend. Install requirements.txt (torch, transformers, accelerate) "
            "or use the llama.cpp server backend (generate-llamacpp).\n\n"
            f"Original error: {e}"
        )

    prompt = _read_prompt(args)

    out_path = Path(args.out) if args.out else Path("logs") / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json"

    decode = DecodeConfig(
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        min_p=0.0,
        repetition_penalty=1.0,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    log = generate_with_entropy(
        model_id=args.model,
        prompt=prompt,
        out_path=out_path,
        system=args.system,
        chat=args.chat,
        device=args.device,
        dtype=args.dtype,
        trust_remote_code=not args.no_trust_remote_code,
        decode=decode,
    )
    print(f"Wrote log: {out_path}")
    print("\n---\nResponse:\n")
    print(log.get("response_text", ""))



def cmd_generate_llamacpp(args: argparse.Namespace) -> None:
    try:
        from .backends.llamacpp_backend import LlamaCppConfig, generate_with_entropy_lower_bound
    except ModuleNotFoundError as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency for llama.cpp server mode. Install requirements.txt (requests).\n\n"
            f"Original error: {e}"
        )

    prompt = _read_prompt(args)
    out_path = Path(args.out) if args.out else Path("logs") / f"run_llamacpp_{time.strftime('%Y%m%d_%H%M%S')}.json"

    cfg = LlamaCppConfig(
        base_url=args.base_url.rstrip("/"),
        model=args.model or "",
        endpoint=args.endpoint,
        kmax=args.kmax,
        timeout=args.timeout,
        max_tokens=args.max_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        repeat_penalty=1.0,
        stream=True,
        system=args.system,
    )
    log = generate_with_entropy_lower_bound(cfg=cfg, prompt=prompt, out_path=out_path)
    print(f"Wrote log: {out_path}")
    print("\n---\nResponse:\n")
    print(log.get("response_text", ""))


def cmd_analyze(args: argparse.Namespace) -> None:
    from .analyze import load_log, token_stats, summarize_range

    path = Path(args.log)
    log = load_log(path)

    summary = summarize_range(log, args.start, args.end)
    stats = token_stats(
        log,
        start=args.start,
        end=args.end,
        min_count=args.min_count,
        top_n=args.top_n,
    )
    out = {"summary": summary, "token_stats": stats}
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote analysis JSON: {args.json_out}")
    else:
        print(json.dumps(out, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="gosset",
        description="Generate completions and log per-token Shannon entropy; visualize in a small web UI."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a completion and write a JSON log.")
    g.add_argument("--model", required=True, help="HF model id or local path, e.g. gpt-oss-20b")
    g.add_argument("--prompt", help="Prompt text. If omitted, use --prompt-file or stdin.")
    g.add_argument("--prompt-file", help="Path to prompt text file.")
    g.add_argument("--system", help="Optional system prompt (only used with --chat).")
    g.add_argument("--chat", action="store_true", help="Use tokenizer.apply_chat_template if available.")
    g.add_argument("--out", help="Output JSON path (default: logs/run_YYYYMMDD_HHMMSS.json).")
    g.add_argument("--max-new-tokens", type=int, default=1024)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    g.add_argument("--dtype", default="auto", help="auto|float16|bfloat16|float32")
    g.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code.")
    g.set_defaults(func=cmd_generate)


    l = sub.add_parser("generate-llamacpp", help="Generate via a llama.cpp server and log an entropy lower-bound (top-k only).")
    l.add_argument("--base-url", required=True, help="Base URL, e.g. http://localhost:8080")
    l.add_argument("--endpoint", choices=["chat", "completion"], default="chat", help="Which endpoint to use.")
    l.add_argument("--model", default="", help="Model name (for /v1/chat/completions).")
    l.add_argument("--prompt", help="Prompt text. If omitted, use --prompt-file or stdin.")
    l.add_argument("--prompt-file", help="Path to prompt text file.")
    l.add_argument("--system", help="Optional system prompt (chat endpoint only).")
    l.add_argument("--out", help="Output JSON path.")
    l.add_argument("--kmax", type=int, default=512, help="Top-k candidates requested from server (n_probs/top_logprobs).")
    l.add_argument("--max-tokens", type=int, default=2048)
    l.add_argument("--timeout", type=float, default=600.0)
    l.set_defaults(func=cmd_generate_llamacpp)

    a = sub.add_parser("analyze", help="Compute top tokens by average entropy from a log.")
    a.add_argument("--log", required=True, help="Path to a JSON log produced by 'generate'.")
    a.add_argument("--min-count", type=int, default=5)
    a.add_argument("--top-n", type=int, default=50)
    a.add_argument("--start", type=int, default=None, help="Start token index (inclusive).")
    a.add_argument("--end", type=int, default=None, help="End token index (inclusive).")
    a.add_argument("--json-out", help="Write analysis to this path instead of stdout.")
    a.set_defaults(func=cmd_analyze)

    return ap


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
