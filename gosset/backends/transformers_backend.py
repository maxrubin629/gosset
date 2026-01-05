from __future__ import annotations

import math
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
        # `device_map="auto"` requires accelerate (transformers treats it as an optional dep).
        try:  # pragma: no cover
            import accelerate  # noqa: F401
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "device='auto' requires the 'accelerate' package. Install it (pip install accelerate) "
                "or pass --device cpu|cuda|mps.\n\n"
                f"Original error: {e}"
            )
        kwargs["device_map"] = "auto"
        # Helps memory usage for large models.
        kwargs.setdefault("low_cpu_mem_usage", True)

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
) -> tuple[torch.Tensor, bool]:
    used_chat_template = False
    if chat and hasattr(tokenizer, "apply_chat_template"):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        used_chat_template = True
    else:
        enc = tokenizer(prompt, return_tensors="pt")
    return enc["input_ids"], used_chat_template


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

    # Prefer global seeding for portability (MPS, CPU, etc.). For CUDA we also pass an explicit generator
    # matching the logits device (see below).
    torch.manual_seed(int(decode.seed))

    input_ids, used_chat_template = build_prompt_input_ids(tokenizer, prompt, system=system, chat=chat)
    input_ids = input_ids.to(inference_device)

    generated_ids: List[int] = []
    steps: List[TokenStep] = []

    # Prefill
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
        past = out.past_key_values
        next_logits = out.logits[:, -1, :].squeeze(0)  # [V]

    # Create a generator that matches the device where sampling occurs (CUDA requires this).
    # For non-CUDA backends, passing a generator can be unsupported; seeding above is usually enough.
    gen: Optional[torch.Generator] = None
    gen_device: Optional[torch.device] = None
    if next_logits.device.type == "cuda":
        gen = torch.Generator(device=next_logits.device)
        gen.manual_seed(int(decode.seed))
        gen_device = next_logits.device
    elif next_logits.device.type == "cpu":
        gen = torch.Generator()
        gen.manual_seed(int(decode.seed))
        gen_device = next_logits.device

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)

    # Some configs expose multiple EOS ids.
    eos_ids: Optional[set[int]]
    if eos_id is None:
        eos_ids = None
    elif isinstance(eos_id, (list, tuple, set)):
        eos_ids = {int(x) for x in eos_id}
    else:
        eos_ids = {int(eos_id)}

    t0 = time.time()
    for i in range(decode.max_new_tokens):
        # If the backend/device_map changes where logits live, refresh the generator.
        if next_logits.device.type in ("cuda", "cpu") and gen_device != next_logits.device:
            if next_logits.device.type == "cuda":
                gen = torch.Generator(device=next_logits.device)
                gen.manual_seed(int(decode.seed))
            else:
                gen = torch.Generator()
                gen.manual_seed(int(decode.seed))
            gen_device = next_logits.device
        elif next_logits.device.type not in ("cuda", "cpu"):
            gen = None
            gen_device = None

        ent_nats = float(shannon_entropy_from_logits(next_logits, base="e").item())
        ent_bits = ent_nats / math.log(2.0)

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

        if eos_ids is not None and token_id in eos_ids:
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
        "chat_template_used": bool(used_chat_template),
        "chat_template_requested": bool(chat),
        "decode": asdict(decode),
        "timing": {"seconds": dt, "tokens_per_second": (len(steps) / dt) if dt > 0 else None},
        "steps": [asdict(s) for s in steps],
        "response_text": response_text,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log
