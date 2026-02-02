# gpt-oss entropy tool (gosset)

A small repo that:

1) streams a full response from a llama.cpp server (intended: `gpt-oss-20b`) at **T=1, top_k=0, top_p=1, min_p=0, repetition_penalty=1**  
2) logs **per-step Shannon entropy** (both nats and bits) for the next-token distribution (lower bound from top-k)  
3) provides a lightweight viewer that renders each generated token with a green↔red entropy heatmap and a two-click token-range selector.

<img width="1128" height="831" alt="Screenshot 2026-01-05 at 7 44 36 AM" src="https://github.com/user-attachments/assets/488ac355-ee3d-4c7c-b679-7ee3808c5111" />

## What "entropy per token" means here

For each generation step *t*, we compute:

*H_t = -∑ p_t(i) log p_t(i)*

where *p_t* is the model's next-token distribution at that step (temperature=1 by default).
This is entropy of the **distribution at position t**, not a property of the sampled token itself.

## Install

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

Notes:
* Generation is via a llama.cpp server (see `generate-llamacpp` below).
* `gosset analyze` and the `viewer/` UI can be used on existing JSON logs without running a server.

## Generate a log

### llama.cpp server mode

If you're running `gpt-oss-20b` behind a llama.cpp server, you can log an **entropy lower bound**
computed from the returned top-k probabilities (because the full softmax over the entire vocab is not available).

Example server command:

```bash
llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 0 --jinja -ub 2048 -b 2048
```

```bash
python -m gosset generate-llamacpp \
  --base-url http://localhost:8080 \
  --endpoint chat \
  --model gpt-oss-20b \
  --kmax 512 \
  --prompt "Write a short proof sketch for the AM-GM inequality." \
  --out logs/llamacpp.json
```

The logged `entropy_*` values in this mode are explicitly marked as a **lower bound**.

## Analyze a log (top tokens by average entropy)

```bash
python -m gosset analyze --log logs/llamacpp.json --min-count 5 --top-n 50
```

You can restrict the analysis to a token index range:

```bash
python -m gosset analyze --log logs/llamacpp.json --start 120 --end 420 --min-count 3
```

## View (token heatmap + range selection)

The viewer is a static site in `viewer/`.

```bash
cd viewer
python -m http.server 8000
```

Then open the served page and load a JSON log.

Features:
* token background: green (low entropy) ↔ red (high entropy); transparent near the middle band
* **range selection**: click token A (start), click token B (end)
* "Top tokens by average entropy" for (a) the full run and (b) your selection

## File format (log JSON)

The generator writes a single JSON file:

```json
{
  "schema_version": 1,
  "backend": "llamacpp_topk_lower_bound",
  "model_id": "...",
  "kmax": 512,
  "note": "entropy is a LOWER BOUND when computed from top-k only",
  "decode": { "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "max_new_tokens": 16384, "seed": 0 },
  "steps": [
    { "index": 0, "token_id": 123, "token": "Hello", "entropy_nats": 2.34, "entropy_bits": 3.37, "mass_observed": 0.91, "top_logprobs": [] }
  ],
  "response_text": "..."
}
```

Notes for llama.cpp logs:
* Top-level `kmax` is recorded, and `decode.seed` is always set (random if not provided).
* Each step includes `token_id`, `is_prompt` (currently false for generated tokens), and `top_logprobs` (chosen token + up to 5 alternatives).

## Future: MLX LM backend outline (not implemented)

Planned steps for adding an MLX LM backend:
* Verify MLX LM API support for per-token logprobs (or full logits) in streaming generation.
* If only top-k logprobs are available, compute an entropy lower bound like the llama.cpp backend.
* Add `gosset/backends/mlx_backend.py` to stream tokens and log `entropy_nats`/`entropy_bits` (and `mass_observed`/`top_logprobs` if top-k only).
* Add `generate-mlx` subcommand with minimal flags: `--model`, `--prompt/--prompt-file`, `--out`, `--max-new-tokens`, `--seed`.
* Add `requirements-mlx.txt` with `mlx-lm` and document Apple Silicon-only usage in the README.
* Ensure MLX logs follow the existing schema fields used by `viewer/app.js`.
