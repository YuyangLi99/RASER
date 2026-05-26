"""Shared helpers for calling reasoning-capable chat models on the KIT toolbox.

Reasoning-model output failure modes we explicitly handle:

  1. Inline <think>...</think> closed normally  → strip the block.
  2. Inline <think> opened but never closed (truncated by max_tokens)
     → mark as `truncated_reasoning`, retry once with larger budget.
  3. content == "" with reasoning_content present (Qwen-style separate field)
     and finish_reason == "length" → retry once with larger budget.
  4. content == "" with finish_reason == "stop" (genuine empty) → return "".

Per-provider max_tokens scaling (output budget includes hidden reasoning):
  - gpt-oss-* :  ×3 of caller-requested max_tokens (reasoning is short).
  - qwen3*    :  fixed floor of 8000 (reasoning chains often 3–6k tokens).
  - minimax*  :  ×6, floor 6000 (inline <think> + answer).
  - mistral-small-4, gpt-5, o3, o4 : ×3 (conservative default).
  - non-reasoning models : unchanged.
"""

import re

_REASONING_TAGS = (
    "gpt-oss", "qwen3", "minimax", "mistral-small-4",
    "gpt-5", "o3", "o4",
)

_THINK_OPEN_RE = re.compile(r"<think>", re.IGNORECASE)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def is_reasoning_model(name: str) -> bool:
    n = (name or "").lower()
    return any(tag in n for tag in _REASONING_TAGS)


def _provider_max_tokens(model: str, requested: int) -> int:
    """Return the effective max_tokens for a single LLM call.

    Each model has both an output multiplier (some need more headroom for
    reasoning) and an absolute output cap (to stay within the model's
    context length, especially for self-hosted vLLM models with smaller
    contexts).
    """
    n = (model or "").lower()
    # Self-hosted vLLM endpoints — cap to avoid input+output exceeding ctx
    if "phi-4-mini" in n or "phi4-mini" in n:
        return min(2048, max(2048, requested * 2))   # Phi-4-mini 8192 ctx
    if "qwen3-8b" in n:
        return min(4096, max(4096, requested * 3))   # Qwen3-8B 32K ctx, but KV cache limits output
    if "qwen3.5-9b" in n or "qwen3_5-9b" in n:
        return min(4096, max(4096, requested * 3))
    # KIT-served reasoning / large models
    if "qwen3.5-397" in n or "qwen3_5-397" in n:
        return max(8000, requested * 6)
    if "qwen3" in n:                                  # remaining Qwen3 variants
        return max(8000, requested * 6)
    if "minimax" in n:
        return max(6000, requested * 6)
    if any(tag in n for tag in ("gpt-oss", "mistral-small-4", "gpt-5", "o3", "o4")):
        return requested * 3
    return requested


def _provider_timeout(model: str, requested) -> int:
    if requested is not None:
        return requested
    n = (model or "").lower()
    if "qwen3" in n or "minimax" in n:
        return 300
    if is_reasoning_model(model):
        return 180
    return 60


def has_unclosed_think(text: str) -> bool:
    """True if a <think> block was opened but never closed (truncated reasoning)."""
    if not text:
        return False
    opens = len(_THINK_OPEN_RE.findall(text))
    closes = len(_THINK_CLOSE_RE.findall(text))
    return opens > closes


def strip_think_block(text: str) -> str:
    """Remove all closed <think>...</think> blocks. Leaves unclosed think intact."""
    if not text:
        return ""
    return _THINK_BLOCK_RE.sub("", text).strip()


def _extract_clean(resp) -> tuple:
    """Return (content, finish_reason, reasoning_content)."""
    choice = resp.choices[0]
    msg = choice.message
    content = msg.content or ""
    finish = getattr(choice, "finish_reason", None) or ""
    reasoning = getattr(msg, "reasoning_content", None) or ""
    return content, finish, reasoning


def _classify_failure(content: str, finish_reason: str, reasoning: str) -> str:
    """Return a short tag describing why this response is unusable, or '' if OK."""
    if has_unclosed_think(content):
        return "truncated_think"
    if not content.strip():
        if finish_reason == "length":
            return "empty_length_truncated"
        if reasoning.strip():
            return "empty_with_reasoning"
        return "empty_clean"
    return ""


def call_chat(client, model: str, messages: list,
              max_tokens: int, temperature: float = 0.0,
              timeout: int = None, _retry: bool = False) -> tuple:
    """Call chat.completions.create with reasoning-model adjustments.

    Returns (content_str, token_count). Never raises; logs errors and returns
    ("", 0) on failure. On retry-eligible failures, retries once with 2× budget.
    """
    effective_max = _provider_max_tokens(model, max_tokens)
    if _retry:
        effective_max = effective_max * 2
    effective_timeout = _provider_timeout(model, timeout)
    if _retry:
        effective_timeout = max(effective_timeout, 480)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=effective_timeout,
            max_tokens=effective_max,
        )
    except Exception as e:
        print(f"  [call_chat ERR] {type(e).__name__}: {str(e)[:120]}", flush=True)
        if not _retry:
            return call_chat(client, model, messages,
                             max_tokens=max_tokens, temperature=temperature,
                             timeout=timeout, _retry=True)
        return "", 0

    # KIT gateway sometimes returns a 200 with a None body on transient 5xx;
    # treat as a recoverable failure and retry once instead of crashing.
    if resp is None or not getattr(resp, "choices", None):
        print(f"  [call_chat ERR] empty/null response (model={model})", flush=True)
        if not _retry:
            return call_chat(client, model, messages,
                             max_tokens=max_tokens, temperature=temperature,
                             timeout=timeout, _retry=True)
        return "", 0

    content, finish, reasoning = _extract_clean(resp)
    tokens = resp.usage.total_tokens if resp.usage else 0
    fail = _classify_failure(content, finish, reasoning)

    if fail in ("truncated_think", "empty_length_truncated", "empty_with_reasoning") and not _retry:
        print(f"  [call_chat RETRY] reason={fail} model={model} "
              f"max_tokens {effective_max}->{effective_max*2}", flush=True)
        return call_chat(client, model, messages,
                         max_tokens=max_tokens, temperature=temperature,
                         timeout=timeout, _retry=True)

    if fail == "truncated_think":
        # Retry already attempted; salvage by stripping the open <think> tail.
        idx = content.lower().rfind("<think>")
        content = content[:idx].strip() if idx >= 0 else ""
    else:
        content = strip_think_block(content).strip()

    return content, tokens
