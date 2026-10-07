#!/usr/bin/env python3
"""Patch vLLM 0.13.0 FlashInfer planning for CompressLM code masks."""

from pathlib import Path
import sys


MARKER = "# ABRIDGEKV_COMPRESSLM_CUSTOM_MASK"


def replace_once(source: str, old: str, new: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one FlashInfer patch anchor, found {count}: "
            f"{old[:80]!r}"
        )
    return source.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_stock_flashinfer.py PATH")
    path = Path(sys.argv[1])
    source = path.read_text()
    if MARKER in source:
        print(f"[overlay] FlashInfer CompressLM patch already present: {path}")
        return

    source = replace_once(
        source,
        """        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu

        if self.dcp_world_size > 1:
""",
        f"""        qo_indptr_cpu = common_attn_metadata.query_start_loc_cpu
        {MARKER}
        causal = common_attn_metadata.causal
        custom_mask = getattr(
            common_attn_metadata, "compresslm_custom_mask", None
        )
        if custom_mask is not None and num_decodes and not getattr(
            common_attn_metadata, "compresslm_custom_mask_prefill_only", False
        ):
            # FlashInfer's decode wrapper has no custom-mask API. Route q=1
            # requests through the paged prefill wrapper instead (unless the
            # mask covers only the prefill slice: mixed CompressLM code steps).
            num_prefills = num_reqs
            num_prefill_tokens = num_actual_tokens
            num_decodes = 0
            num_decode_tokens = 0
        if (not causal or custom_mask is not None) and self.dcp_world_size > 1:
            raise RuntimeError(
                "CompressLM code attention does not support DCP"
            )

        if self.dcp_world_size > 1:
""",
    )
    source = replace_once(
        source,
        "        use_cascade = common_prefix_len > 0\n",
        "        use_cascade = common_prefix_len > 0 and custom_mask is None\n",
    )
    source = replace_once(
        source,
        """        decode_use_trtllm = (
            self.use_trtllm_decode_attention and self.dcp_world_size <= 1
        )
""",
        """        if not causal or custom_mask is not None:
            prefill_use_trtllm = False
        decode_use_trtllm = (
            self.use_trtllm_decode_attention and self.dcp_world_size <= 1
        )
""",
    )
    source = replace_once(
        source,
        """            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
        )
""",
        """            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
        )
        attn_metadata.causal = causal
""",
    )
    source = replace_once(
        source,
        """                paged_kv_indptr_cpu = paged_kv_indptr_cpu[prefill_start:]

                # Recompute max_q_len for the slice of requests we are using
""",
        """                paged_kv_indptr_cpu = paged_kv_indptr_cpu[prefill_start:]

                qo_indptr_plan = qo_indptr_cpu
                paged_kv_indptr_plan = paged_kv_indptr_cpu
                paged_kv_last_page_len_plan = (
                    paged_kv_last_page_len_cpu[prefill_start:]
                )
                if custom_mask is not None:
                    # FlashInfer 0.5.3 packs custom masks on-device and requires
                    # every indptr used to construct mask_indptr to be CUDA.
                    qo_indptr_plan = qo_indptr_plan.to(
                        self.device, non_blocking=True
                    )
                    paged_kv_indptr_plan = paged_kv_indptr_plan.to(
                        self.device, non_blocking=True
                    )
                    paged_kv_last_page_len_plan = (
                        paged_kv_last_page_len_plan.to(
                            self.device, non_blocking=True
                        )
                    )

                # Recompute max_q_len for the slice of requests we are using
""",
    )
    source = replace_once(
        source,
        """                            qo_indptr_cpu,
                            paged_kv_indptr_cpu,
                            paged_kv_indices,
                            paged_kv_last_page_len_cpu[prefill_start:],
""",
        """                            qo_indptr_plan,
                            paged_kv_indptr_plan,
                            paged_kv_indices,
                            paged_kv_last_page_len_plan,
""",
    )
    source = replace_once(
        source,
        """                            self.head_dim,
                            self.page_size,
                            causal=True,
                            sm_scale=self.sm_scale,
""",
        """                            self.head_dim,
                            self.page_size,
                            custom_mask=custom_mask,
                            causal=causal,
                            sm_scale=self.sm_scale,
""",
    )
    source = replace_once(
        source,
        "                    assert prefill_wrapper._causal\n",
        "                    assert prefill_wrapper._causal == attn_metadata.causal\n",
    )

    path.write_text(source)


MARKER_FA2 = "# ABRIDGEKV_COMPRESSLM_FA2_PREFILL"


def patch_fa2_prefill(path: Path) -> None:
    """FlashInfer resolves backend="auto" ONCE, at the first plan() (the prompt prefill, no mask) -> fa3 on sm90; the
    FA3 kernels do not support custom masks, so the first masked code step fails with "operation not supported" on
    H100. Pin the paged prefill wrapper to fa2 whenever CompressLM custom masks can appear."""
    source = path.read_text()
    if MARKER_FA2 in source:
        print(f"[overlay] FlashInfer fa2-prefill patch already present: {path}")
        return
    source = replace_once(
        source,
        """                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(), get_kv_cache_layout()
                )
""",
        f"""                {MARKER_FA2}
                _clm = getattr(getattr(self, "vllm_config", None), "compresslm_config", None)
                _needs_mask = (
                    _clm is not None
                    and getattr(_clm, "enable", False)
                    and (
                        _clm.uses_layer_quotas
                        or _clm.uses_selective_history
                        or (bool(getattr(_clm, "code_excluded_tail", 0)) and not _clm.uses_packed_kv)
                    )
                )
                self._prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
                    self._get_workspace_buffer(),
                    get_kv_cache_layout(),
                    backend="fa2" if _needs_mask else "auto",
                )
""",
    )
    path.write_text(source)
    print(f"[overlay] FlashInfer fa2-prefill patch applied: {path}")
    print(f"[overlay] patched FlashInfer CompressLM masks: {path}")


if __name__ == "__main__":
    main()
    patch_fa2_prefill(Path(sys.argv[1]))
