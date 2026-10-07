"""Quota-schedule helpers shared by the vLLM evaluators (scripts/ is on sys.path when they run)."""
import json
from pathlib import Path


def load_quota_rows(path: str) -> tuple[tuple[int, ...], ...]:
    """36x8 (layer x KV head) quota rows from a schedule file: a 2D array, or an object with 'quotas'/'packed_kv_quotas'."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("packed_kv_quotas", payload.get("quotas"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{path}: expected a non-empty 2D quota array or an object with 'quotas'")
    return tuple(tuple(int(v) for v in row) for row in payload)


def load_layer_quotas(path: str) -> tuple[int, ...]:
    """Per-layer quotas for masked serving on the stock cache: every layer's row must be uniform across KV heads
    (tensor-parallel serving cannot give heads of one layer different quotas)."""
    rows = load_quota_rows(path)
    quotas = []
    for layer, row in enumerate(rows):
        if len(set(row)) != 1:
            raise ValueError(f"{path}: layer {layer} has per-head quotas {sorted(set(row))}; per-layer serving needs one quota per layer")
        quotas.append(row[0])
    return tuple(quotas)
