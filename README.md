# gpt-oss entropy kit

A small repo that:

1) generates a full response from a causal LM (intended: `gpt-oss-20b`) at **T=1, top_k=0, top_p=1, min_p=0, repetition_penalty=1**  
2) logs **per-step Shannon entropy** (both nats and bits) for the next-token distribution  
3) provides a lightweight viewer that renders each generated token with a green↔red entropy heatmap and a two-click token-range selector.

## What “entropy per token” means here

For each generation step *t*, we compute:

*H_t = -∑ p_t(i) log p_t(i)*

where *p_t* is the model’s next-token distribution at that step (temperature=1 by default).
This is entropy of the **distribution at position t**, not a property of the sampled token itself.

## Install

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Generate a log

### Plain prompt

```bash
python -m gosset generate \
  --model gpt-oss-20b \
  --prompt "Explain varentropy in one paragraph." \
  --out logs/varentropy.json \
  --max-new-tokens 1024 \
  --seed 0
```

### Chat template (if the tokenizer supports it)

```bash
python -m gosset generate \
  --model gpt-oss-20b \
  --chat \
  --system "You are a precise assistant." \
  --prompt "Solve: what is 1+1 in base 2?" \
  --out logs/base2.json
```

Device/dtype knobs (optional):

```bash
python -m gosset generate --model gpt-oss-20b --prompt "..." --device mps --dtype float16
```

Notes:
* `--device auto` uses `device_map="auto"` (requires `accelerate`).
* If you’re on Apple Silicon, `--device mps` is usually the simplest path.

## Analyze a log (top tokens by average entropy)

```bash
python -m gosset analyze --log logs/base2.json --min-count 5 --top-n 50
```

You can restrict the analysis to a token index range:

```bash
python -m gosset analyze --log logs/base2.json --start 120 --end 420 --min-count 3
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
* “Top tokens by average entropy” for (a) the full run and (b) your selection

## File format (log JSON)

The generator writes a single JSON file:

```json
{
  "schema_version": 1,
  "backend": "transformers",
  "model_id": "...",
  "decode": { "temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0, "repetition_penalty": 1.0, "max_new_tokens": 1024, "seed": 0 },
  "steps": [
    { "index": 0, "token_id": 123, "token": "Hello", "entropy_nats": 2.34, "entropy_bits": 3.37 }
  ],
  "response_text": "..."
}
```

## llama.cpp server mode

If you’re running `gpt-oss-20b` behind a llama.cpp server, you can log an **entropy lower bound**
computed from the returned top-k probabilities (because the full softmax over the entire vocab is not available).

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
