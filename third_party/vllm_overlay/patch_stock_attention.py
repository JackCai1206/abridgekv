#!/usr/bin/env python3
"""Patch vLLM 0.13.0 Attention to consume packed CompressLM KV pages."""

from pathlib import Path
import sys


MARKER = "# ABRIDGEKV_COMPRESSLM_PACKED_KV"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_stock_attention.py PATH")
    path = Path(sys.argv[1])
    source = path.read_text()
    if MARKER in source:
        print(f"[overlay] packed-KV attention patch already present: {path}")
        return

    anchor = """            return output.view(-1, hidden_size)
        else:
"""
    replacement = f"""            {MARKER}
            if hasattr(self, "_compresslm_packed_sidecar"):
                forward_context = get_forward_context()
                packed_attn_metadata = forward_context.attn_metadata
                if isinstance(packed_attn_metadata, dict):
                    packed_attn_metadata = packed_attn_metadata[self.layer_name]
                packed_kv_cache = self.kv_cache[forward_context.virtual_engine]
                from vllm.v1.worker.compresslm_packed_kv import (
                    maybe_override_attention,
                )
                maybe_override_attention(
                    self,
                    query,
                    packed_kv_cache,
                    packed_attn_metadata,
                    output,
                )
            return output.view(-1, hidden_size)
        else:
"""
    count = source.count(anchor)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one vLLM Attention patch anchor, found {count}"
        )
    path.write_text(source.replace(anchor, replacement, 1))
    print(f"[overlay] patched CompressLM packed-KV attention: {path}")


if __name__ == "__main__":
    main()
