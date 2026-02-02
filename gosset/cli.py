from __future__ import annotations

import argparse
import sys
import json
import time
from pathlib import Path
from typing import Optional


# NOTE: Backends are imported lazily inside subcommands so that lightweight
# operations like `gosset analyze` don't require installing requests until needed.


def _read_prompt(args: argparse.Namespace) -> str:
    if args.prompt is not None:
        return args.prompt
    if args.prompt_file is not None:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    # stdin
    if sys.stdin.isatty():
        raise SystemExit("No prompt provided. Use --prompt, --prompt-file, or pipe via stdin.")
    return sys.stdin.read()


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
        seed=args.seed,
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


    l = sub.add_parser("generate-llamacpp", help="Generate via a llama.cpp server and log an entropy lower-bound (top-k only).")
    l.add_argument("--base-url", required=True, help="Base URL, e.g. http://localhost:8080")
    l.add_argument("--endpoint", choices=["chat", "completion"], default="chat", help="Which endpoint to use.")
    l.add_argument("--model", default="", help="Model name (for /v1/chat/completions).")
    l.add_argument("--prompt", help="Prompt text. If omitted, use --prompt-file or stdin.")
    l.add_argument("--prompt-file", help="Path to prompt text file.")
    l.add_argument("--system", help="Optional system prompt (chat endpoint only).")
    l.add_argument("--out", help="Output JSON path.")
    l.add_argument("--kmax", type=int, default=512, help="Top-k candidates requested from server (n_probs/top_logprobs).")
    l.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation. If omitted, gosset picks a random seed and records it in the JSON log.",
    )
    l.add_argument("--max-tokens", type=int, default=16384)
    l.add_argument("--timeout", type=float, default=600.0)
    l.set_defaults(func=cmd_generate_llamacpp)

    a = sub.add_parser("analyze", help="Compute top tokens by average entropy from a log.")
    a.add_argument("--log", required=True, help="Path to a JSON log produced by 'generate-llamacpp'.")
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
