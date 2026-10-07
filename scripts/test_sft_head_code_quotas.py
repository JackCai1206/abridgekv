#!/usr/bin/env python3
"""Focused CPU checks for SFT quota parsing, rebucketing, and mask semantics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abridgekv.sft_quota_utils import (
    load_head_code_quotas,
    rebucket_pairs,
)


def test_schedule_loading() -> None:
    expected = ((1, 4), (3, 2))
    fixture = ROOT / "scripts/fixtures/head_code_quotas_2x2.json"
    assert load_head_code_quotas(
        fixture,
        num_layers=2,
        num_kv_heads=2,
        k_max=4,
    ) == expected
    assert load_head_code_quotas(
        json.dumps([[1, 4], [3, 2]]),
        num_layers=2,
        num_kv_heads=2,
        k_max=4,
    ) == expected
    for key in ("quotas", "head_code_quotas", "packed_kv_quotas"):
        assert load_head_code_quotas(
            json.dumps({key: [[1, 4], [3, 2]]}),
            num_layers=2,
            num_kv_heads=2,
            k_max=4,
        ) == expected
    analysis = {"budgets": {"2": {"train_K_lh": [[1, 2], [2, 3]]}}}
    assert load_head_code_quotas(
        json.dumps(analysis),
        budget="2",
        num_layers=2,
        num_kv_heads=2,
        k_max=4,
    ) == ((1, 2), (2, 3))
    try:
        load_head_code_quotas(
            json.dumps(analysis),
            num_layers=2,
            num_kv_heads=2,
            k_max=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("quota-analysis JSON accepted without a budget")
    assert load_head_code_quotas(
        "uniform:2",
        num_layers=2,
        num_kv_heads=2,
        k_max=4,
    ) == ((2, 2), (2, 2))
    for invalid in (
        [[1, 2]],
        [[1], [2]],
        [[0, 1], [1, 1]],
        [[5, 1], [1, 1]],
    ):
        try:
            load_head_code_quotas(
                json.dumps(invalid),
                num_layers=2,
                num_kv_heads=2,
                k_max=4,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid schedule accepted: {invalid}")


def test_c_rebucketing() -> None:
    pairs = [
        {"prompt": [1, 2, 3], "response": [4, 5], "n_blocks": 9},
        {"prompt": [1] * 7, "response": [2] * 2, "n_blocks": 1},
    ]
    rebucketed = rebucket_pairs(pairs, 4)
    assert [item["n_blocks"] for item in rebucketed] == [2, 3]
    assert [item["n_blocks"] for item in pairs] == [9, 1]
    assert rebucketed[0] is not pairs[0]
    assert rebucketed[0]["prompt"] is pairs[0]["prompt"]


def materialize(mask_mod, batches, heads, sequence_length, torch):
    result = torch.empty(
        batches,
        heads,
        sequence_length,
        sequence_length,
        dtype=torch.bool,
    )
    for batch in range(batches):
        for head in range(heads):
            for query in range(sequence_length):
                for key in range(sequence_length):
                    result[batch, head, query, key] = bool(
                        mask_mod(
                            torch.tensor(batch),
                            torch.tensor(head),
                            torch.tensor(query),
                            torch.tensor(key),
                        )
                    )
    return result


def test_torch_masks(torch) -> None:
    from transformers import LlamaConfig, LlamaForCausalLM

    from abridgekv.compress_forward import CompressLM

    config = LlamaConfig(
        vocab_size=67,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    quotas = ((1, 3), (4, 2))
    model = CompressLM(
        LlamaForCausalLM(config).float(),
        C=4,
        N=3,
        K=4,
        L=2,
        pause_xattn="full",
        head_code_quotas=quotas,
    ).eval()
    plain = CompressLM(
        LlamaForCausalLM(config).float(),
        C=4,
        N=3,
        K=4,
        L=2,
        pause_xattn="full",
    ).eval()
    assert set(model.state_dict()) == set(plain.state_dict())
    assert model.query_head_code_quotas.tolist() == [
        [1, 1, 3, 3],
        [4, 4, 2, 2],
    ]
    base_mask = object()
    layer_masks = [object(), object()]
    model._quota_mask_dispatch = {
        id(base_mask): (base_mask, layer_masks)
    }
    _, hooked_kwargs = model._quota_attention_mask_hook(1)(
        None,
        (),
        {"attention_mask": base_mask},
    )
    assert hooked_kwargs["attention_mask"] is layer_masks[1]

    base_mod, sequence_length = model._mask_mod("cpu", False)
    base = materialize(base_mod, 1, 4, sequence_length, torch)
    per = model.C + model.L * model.K
    slots = torch.full((sequence_length,), -1, dtype=torch.long)
    within = torch.arange(sequence_length) % per
    is_code = (within >= model.C) & (
        within < model.C + model.L * model.K
    )
    slots[is_code] = (within[is_code] - model.C) % model.K

    for layer in range(model.num_layers):
        quota_mod, quota_length = model._mask_mod(
            "cpu",
            False,
            layer,
        )
        assert quota_length == sequence_length
        actual = materialize(
            quota_mod,
            1,
            model.num_attention_heads,
            sequence_length,
            torch,
        )
        expanded = model.query_head_code_quotas[layer]
        key_allowed = (
            (slots < 0).view(1, 1, 1, -1)
            | (
                slots.view(1, 1, 1, -1)
                < expanded.view(1, -1, 1, 1)
            )
        )
        expected = base & key_allowed
        assert torch.equal(actual, expected)

    layer0_mod, _ = model._mask_mod("cpu", False, 0)
    q_block1_content = per
    final_pass_start = model.C + model.K
    scalar = lambda value: torch.tensor(value, dtype=torch.int64)
    assert bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(q_block1_content),
        scalar(final_pass_start),
    ))
    assert not bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(q_block1_content),
        scalar(final_pass_start + 1),
    ))
    q_block2_content = 2 * per
    assert bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(q_block2_content),
        scalar(final_pass_start),
    ))
    assert bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(q_block2_content),
        scalar(per + final_pass_start),
    ))
    assert not bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(q_block2_content),
        scalar(final_pass_start + 1),
    ))
    assert not bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(model.C),
        scalar(model.C + 3),
    ))
    assert not bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(final_pass_start),
        scalar(model.C + 2),
    ))
    assert bool(layer0_mod(
        scalar(0),
        scalar(0),
        scalar(model.C + 3),
        scalar(0),
    ))

    prompt_len = torch.tensor([5, 2], dtype=torch.int32)
    raw_base_mod, raw_length = model._mask_rawprompt_mod(
        "cpu",
        prompt_len,
    )
    raw_base = materialize(raw_base_mod, 2, 4, raw_length, torch)
    raw_quota_mod, _ = model._mask_rawprompt_mod(
        "cpu",
        prompt_len,
        0,
    )
    raw_actual = materialize(raw_quota_mod, 2, 4, raw_length, torch)
    expanded = model.query_head_code_quotas[0]
    raw_expected = raw_base & (
        (slots < 0).view(1, 1, 1, -1)
        | (
            slots.view(1, 1, 1, -1)
            < expanded.view(1, -1, 1, 1)
        )
    )
    assert torch.equal(raw_actual, raw_expected)
    assert raw_actual[0, 0, per + model.C, 3]
    assert not raw_actual[0, 0, per + model.C, final_pass_start + 1]

    dense = CompressLM(
        LlamaForCausalLM(config).float(),
        C=4,
        N=2,
        K=4,
        L=2,
        pause_xattn="full",
        head_code_quotas=((4, 4), (4, 4)),
    ).eval()
    dense_base_mod, dense_length = dense._mask_mod("cpu", False)
    dense_base = materialize(dense_base_mod, 1, 4, dense_length, torch)
    for layer in range(dense.num_layers):
        dense_mod, _ = dense._mask_mod("cpu", False, layer)
        assert torch.equal(
            materialize(dense_mod, 1, 4, dense_length, torch),
            dense_base,
        )

    dense.set_N(3)
    assert dense._quota_bm is None and dense._quota_bm_copy is None
    assert dense.N == 3


def main() -> None:
    test_schedule_loading()
    test_c_rebucketing()
    try:
        import torch
    except ModuleNotFoundError:
        print("SFT HEAD-CODE QUOTA PURE-PYTHON PASS; TORCH MASK TESTS SKIPPED")
        return
    test_torch_masks(torch)
    print("SFT HEAD-CODE QUOTA PASS")


if __name__ == "__main__":
    main()
