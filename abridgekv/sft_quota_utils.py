"""Pure-Python helpers for SFT quota schedules and token-length rebucketing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def pair_tokens(item):
    """prompt+response token ids of one SFT pair as a plain Python list, whatever the pickle stored
    (Python lists or numpy int32 arrays; the arrays use 10x less RAM)."""
    return np.asarray(item["prompt"]).tolist() + np.asarray(item["response"]).tolist()


_DIRECT_QUOTA_KEYS = (
    "head_code_quotas",
    "quotas",
    "packed_kv_quotas",
)


def _read_schedule_spec(spec):
    if isinstance(spec, (list, dict)):
        return spec
    text = str(spec).strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        path = Path(text).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"head-code quota schedule not found: {text}")
        return json.loads(path.read_text(encoding="utf-8"))


def _extract_schedule(payload, budget):
    if isinstance(payload, dict):
        for key in _DIRECT_QUOTA_KEYS:
            if key in payload:
                return payload[key]
        if "budgets" in payload:
            budget_key = str(budget).strip()
            if not budget_key:
                raise ValueError(
                    "HEAD_CODE_QUOTA_BUDGET is required for a quota-analysis JSON"
                )
            entry = payload["budgets"].get(budget_key)
            if not isinstance(entry, dict) or "train_K_lh" not in entry:
                raise ValueError(
                    f"head-code quota budget {budget_key!r} is absent"
                )
            return entry["train_K_lh"]
    return payload


def validate_head_code_quotas(
    quotas,
    *,
    num_layers,
    num_kv_heads,
    k_max,
):
    if not isinstance(quotas, (list, tuple)) or not quotas:
        raise ValueError("head-code quota schedule must be a non-empty 2D array")
    if len(quotas) != num_layers:
        raise ValueError(
            "head-code quota schedule must have "
            f"{num_layers} layers, got {len(quotas)}"
        )
    normalized = []
    for layer, row in enumerate(quotas):
        if not isinstance(row, (list, tuple)) or len(row) != num_kv_heads:
            got = (
                len(row)
                if isinstance(row, (list, tuple))
                else type(row).__name__
            )
            raise ValueError(
                f"head-code quota layer {layer} must have {num_kv_heads} "
                f"physical KV heads, got {got}"
            )
        normalized_row = []
        for head, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(
                    f"head-code quota [{layer}][{head}] must be an integer"
                )
            if not 0 < value <= k_max:
                raise ValueError(
                    f"head-code quota [{layer}][{head}]={value} must be in "
                    f"[1, {k_max}]"
                )
            normalized_row.append(value)
        normalized.append(tuple(normalized_row))
    return tuple(normalized)


def load_head_code_quotas(
    spec,
    *,
    budget="",
    num_layers,
    num_kv_heads,
    k_max,
):
    if spec is None or not str(spec).strip():
        return None
    text = str(spec).strip()
    if text.startswith("uniform:"):
        try:
            value = int(text.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(
                f"invalid uniform head-code quota specification: {text!r}"
            ) from exc
        payload = [[value] * num_kv_heads for _ in range(num_layers)]
    else:
        payload = _extract_schedule(_read_schedule_spec(spec), budget)
    return validate_head_code_quotas(
        payload,
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        k_max=k_max,
    )


def rebucket_pairs(pairs, block_size):
    """Return shallow pair copies with n_blocks recomputed from existing token IDs."""
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    rebucketed = []
    for pair in pairs:
        token_count = len(pair["prompt"]) + len(pair["response"])
        item = dict(pair)
        item["n_blocks"] = (token_count + block_size - 1) // block_size
        rebucketed.append(item)
    return rebucketed
