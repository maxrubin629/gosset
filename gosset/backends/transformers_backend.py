from __future__ import annotations

import time
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils import shannon_entropy_from_logits, sample_next_token


@dataclass
class DecodeConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    seed: int = 0


@dataclass
class TokenStep:
    index: int
    token_id: int
    token: str
    entropy_nats: float
    entropy_bits: float


def load_tokenizer(model_id: str, *, trust_remote_code: bool = True) -> Any:
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return tok


def load_model(
    model_id: str,
    *,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
) -> Any:
    kwargs: Dict[str, Any] = {"trust_remote_code": trust_remote_code}

    if dtype != "auto":
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        kwargs["torch_dtype"] = dtype_map.get(dtype, torch.float16)

    if device == "auto":
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = None

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()

    if device != "auto":
        model.to(torch.device(device))

    return model


def _infer_inference_device(model: Any) -> torch.device:
    """Best-effort device to place input tensors on (especially for device_map='auto')."""
    for p in model.parameters():
        if p.device.type != "meta":
            return p.device
    return torch.device("cpu")


def build_prompt_input_ids(
    tokenizer: Any,
    prompt: str,
    *,
    system: Optional[str] = None,
    chat: bool = False,
) -> torch.Tensor:
    if chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    return enc["input_ids"]


def generate_with_entropy(
    *,
    model_id: str,
    prompt: str,
    out_path: Path,
    system: Optional[str] = None,
    chat: bool = False,
    device: str = "auto",
    dtype: str = "auto",
    trust_remote_code: bool = True,
    decode: Optional[DecodeConfig] = None,
) -> Dict[str, Any]:
    """Generate a completion and log per-step entropy.

    Entropy is computed from the next-token distribution at each generation step.
    With the requested settings (T=1, top_k=0, top_p=1, min_p=0, rep_pen=1), this is
    the full softmax distribution at temperature 1.
    """
    decode = decode or DecodeConfig()

    tokenizer = load_tokenizer(model_id, trust_remote_code=trust_remote_code)
    model = load_model(model_id, device=device, dtype=dtype, trust_remote_code=trust_remote_code)

    inference_device = _infer_inference_device(model)

    # torch.Generator must match device for CUDA; for other backends we fall back to CPU generator.
    gen = torch.Generator(device=inference_device) if inference_device.type == "cuda" else torch.Generator()
    gen.manual_seed(decode.seed)

    input_ids = build_prompt_input_ids(tokenizer, prompt, system=system, chat=chat).to(inference_device)

    generated_ids: List[int] = []
    steps: List[TokenStep] = []

    # Prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :].squeeze(0)  # [V]

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)

    t0 = time.time()
    for i in range(decode.max_new_tokens):
        ent_nats = float(shannon_entropy_from_logits(next_logits, base="e").item())
        ent_bits = float(shannon_entropy_from_logits(next_logits, base="2").item())

        token_id, _ = sample_next_token(
            next_logits,
            temperature=decode.temperature,
            top_k=decode.top_k,
            top_p=decode.top_p,
            min_p=decode.min_p,
            repetition_penalty=decode.repetition_penalty,
            generated_token_ids=generated_ids,
            generator=gen,
        )

        token_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        steps.append(TokenStep(index=i, token_id=token_id, token=token_text, entropy_nats=ent_nats, entropy_bits=ent_bits))
        generated_ids.append(token_id)

        if eos_id is not None and token_id == eos_id:
            break

        with torch.no_grad():
            tok = torch.tensor([[token_id]], device=inference_device, dtype=input_ids.dtype)
            out = model(input_ids=tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_logits = out.logits[:, -1, :].squeeze(0)

    dt = time.time() - t0
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    log: Dict[str, Any] = {
        "schema_version": 1,
        "backend": "transformers",
        "model_id": model_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt": prompt,
        "system": system,
        "chat_template_used": bool(chat),
        "decode": asdict(decode),
        "timing": {"seconds": dt, "tokens_per_second": (len(steps) / dt) if dt > 0 else None},
        "steps": [asdict(s) for s in steps],
        "response_text": response_text,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log
