"""Compare greedy CompressLM tokens from the HF reference and vLLM overlay."""

import argparse
import hashlib
import json
import sys
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def parse_layer_spec(value: str) -> tuple[int, ...]:
    layers = set()
    for part in value.split(","):
        part = part.strip().removeprefix("l")
        if not part:
            continue
        if "-" in part:
            start, end = map(int, part.split("-", 1))
            layers.update(range(start, end + 1))
        else:
            layers.add(int(part))
    return tuple(sorted(layers))


def load_packed_kv_schedule(
    path: str,
    budget: int | None,
) -> tuple[tuple[int, ...], ...]:
    if not path:
        return ()
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        direct = payload.get("packed_kv_quotas", payload.get("quotas"))
        if direct is not None:
            payload = direct
        elif "budgets" in payload:
            if budget is None:
                raise ValueError(
                    "--packed-kv-budget is required for a quota-analysis file"
                )
            entry = payload["budgets"].get(str(budget))
            if not isinstance(entry, dict) or "train_K_lh" not in entry:
                raise ValueError(
                    f"packed KV budget {budget} is absent from {path}"
                )
            payload = entry["train_K_lh"]
    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "packed KV schedule must be a non-empty 2D JSON array or an "
            "object containing 'packed_kv_quotas'/'quotas'"
        )
    return tuple(tuple(int(value) for value in row) for row in payload)


def make_prompt(tokenizer, length: int) -> list[int]:
    pattern = tokenizer.encode(
        "Solve this problem carefully and give the final answer.",
        add_special_tokens=True,
    )
    if not pattern:
        raise ValueError("tokenizer produced an empty parity prompt")
    return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]


def write_result(
    path: Path,
    backend: str,
    prompt: list[int],
    output: list[int],
    top_logprobs: list[list[dict[str, float | int]]] | None = None,
) -> None:
    digest = hashlib.sha256(
        ",".join(map(str, output)).encode("ascii")
    ).hexdigest()
    path.write_text(
        json.dumps(
            {
                "backend": backend,
                "prompt_token_ids": prompt,
                "output_token_ids": output,
                "output_sha256": digest,
                "top_logprobs": top_logprobs,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"[compresslm-parity] {backend}: {len(output)} tokens sha256={digest}"
    )


def run_hf(
    args, prompt: list[int]
) -> tuple[list[int], list[list[dict[str, float | int]]] | None]:
    import torch
    from huggingface_hub import snapshot_download

    from abridgekv.gen_compress import CompressGen, _load_compresslm

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    base_model = args.base_model or payload.get("base")
    if not base_model:
        raise ValueError("checkpoint has no base model metadata")
    local_base = snapshot_download(base_model, local_files_only=True)
    clm, _, _, _ = _load_compresslm(
        args.checkpoint, local_base, "cuda", torch.bfloat16
    )
    generator = CompressGen(
        clm,
        "cuda",
        raw_prompt_len=len(prompt) if clm.raw_prompt else 0,
        oracle_history_layers=args.retain_history_layers,
        history_replaces_codes_layers=args.replace_codes_layers,
        capture_hidden=bool(args.debug_hidden_dir),
        head_code_quotas=args.hf_quotas or args.packed_kv_quotas or None,
    )
    output: list[int] = []
    top_logprobs = [] if args.top_logprobs else None
    with torch.inference_mode():
        logits = generator.prefill(prompt)
        for index in range(args.max_tokens):
            if top_logprobs is not None:
                values, token_ids = torch.topk(logits.float(), args.top_logprobs)
                log_normalizer = torch.logsumexp(logits.float(), dim=-1)
                top_logprobs.append(
                    [
                        {
                            "token_id": int(token_id),
                            "logprob": float(value - log_normalizer),
                        }
                        for token_id, value in zip(token_ids, values)
                    ]
                )
            token_id = int(logits.argmax())
            output.append(token_id)
            if index + 1 < args.max_tokens:
                logits = generator.step(token_id)
    if args.debug_hidden_dir:
        debug_dir = Path(args.debug_hidden_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        torch.save(generator.hidden_trace, debug_dir / "hf.pt")
    return output, top_logprobs


def run_vllm(
    args, prompt: list[int]
) -> tuple[list[int], list[list[dict[str, float | int]]] | None]:
    from vllm import LLM, SamplingParams
    from vllm.config.compresslm import CompressLMConfig

    metadata = json.loads(
        (Path(args.model_dir) / "compresslm_config.json").read_text()
    )
    metadata.pop("code_token_ids", None)
    # the export also records provenance (code_rows_of_k_max, ...) that is not a config field
    metadata = {k: v for k, v in metadata.items() if k in CompressLMConfig.__dataclass_fields__ and not k.startswith("_")}
    metadata["retain_full_history_layers"] = args.retain_history_layers
    metadata["history_replaces_codes_layers"] = args.replace_codes_layers
    metadata["packed_kv"] = args.packed_kv
    metadata["packed_kv_k_max"] = metadata["num_code_tokens"]
    metadata["packed_kv_quotas"] = args.packed_kv_quotas
    metadata["layer_code_quotas"] = args.layer_quotas
    metadata["physical_layer_quotas"] = bool(args.layer_quotas) and os.environ.get("COMPRESSLM_PHYSICAL_QUOTAS") == "1"
    config = CompressLMConfig(**metadata)
    physical_quotas = config.uses_physical_layer_quotas
    config.debug = bool(args.top_logprobs)
    # Custom attention masks (per-layer quotas, selective history, the static tail's code-step exclusion on the
    # non-packed path) are honored only by the patched FlashInfer backend; FLASH_ATTN ignores them silently.
    if config.uses_layer_quotas or config.uses_selective_history or (config.code_excluded_tail and not config.uses_packed_kv):
        if args.attention_backend.replace("_", "").upper() != "FLASHINFER":
            print(f"[compresslm-parity] custom masks need FLASHINFER; overriding --attention-backend {args.attention_backend}")
            args.attention_backend = "FLASHINFER"
    llm_kwargs = dict(
        model=args.model_dir,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")),
        enable_prefix_caching=False,
        enforce_eager=args.eager or config.uses_packed_kv or (config.uses_layer_quotas and not physical_quotas and os.environ.get("COMPRESSLM_LAYER_PIECEWISE") != "1"),
        attention_backend=args.attention_backend,
        compresslm_config=config,
    )
    if args.no_cudagraph or config.uses_selective_history or (config.uses_layer_quotas and not physical_quotas):
        llm_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}
    if config.uses_layer_quotas and not physical_quotas and os.environ.get("COMPRESSLM_LAYER_PIECEWISE") == "1":
        llm_kwargs["compilation_config"] = {"cudagraph_mode": "PIECEWISE"}   # attention eager (custom masks), the rest in graphs
    if os.environ.get("VLLM_ASYNC_SCHED") == "1":
        llm_kwargs["async_scheduling"] = True    # async-patched package only (third_party/vllm_overlay_async)
        print("[compresslm-parity] async scheduling ON")
    llm = LLM(**llm_kwargs)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        skip_special_tokens=False,
        bad_words=config.code_tokens,
        logprobs=args.top_logprobs or None,
        extra_args=(
            {"compresslm_prompt_length": len(prompt)}
            if config.uses_selective_history
            else None
        ),
    )
    result = llm.generate(
        [{"prompt_token_ids": prompt}], params, use_tqdm=False
    )
    completion = result[0].outputs[0]
    top_logprobs = None
    if completion.logprobs is not None:
        top_logprobs = [
            [
                {
                    "token_id": int(token_id),
                    "logprob": float(logprob.logprob),
                    "rank": int(logprob.rank),
                }
                for token_id, logprob in sorted(
                    token_logprobs.items(),
                    key=lambda item: item[1].rank,
                )
            ]
            for token_logprobs in completion.logprobs
        ]
    return list(completion.token_ids), top_logprobs


def compare(
    prefix: Path,
    reference_result: str = "",
    candidate_result: str = "",
) -> None:
    reference_path = (
        Path(reference_result)
        if reference_result
        else prefix.with_suffix(".hf.json")
    )
    candidate_path = (
        Path(candidate_result)
        if candidate_result
        else prefix.with_suffix(".vllm.json")
    )
    reference = json.loads(reference_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    if reference["prompt_token_ids"] != candidate["prompt_token_ids"]:
        raise AssertionError("parity prompts differ")
    expected = reference["output_token_ids"]
    actual = candidate["output_token_ids"]
    if expected != actual:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(expected, actual))
                if left != right
            ),
            min(len(expected), len(actual)),
        )
        distribution_detail = ""
        reference_top = reference.get("top_logprobs")
        candidate_top = candidate.get("top_logprobs")
        if (
            reference_top
            and candidate_top
            and mismatch < len(reference_top)
            and mismatch < len(candidate_top)
        ):
            def summarize(rows: list[dict[str, float | int]]) -> str:
                return ", ".join(
                    f"{row['token_id']}:{float(row['logprob']):.6f}"
                    for row in rows[:5]
                )

            distribution_detail = (
                f"; {reference['backend']} top5="
                f"[{summarize(reference_top[mismatch])}], "
                f"{candidate['backend']} top5="
                f"[{summarize(candidate_top[mismatch])}]"
            )
        raise AssertionError(
            "CompressLM greedy parity failed at output token "
            f"{mismatch}: {reference['backend']}="
            f"{expected[mismatch:mismatch + 5]}, "
            f"{candidate['backend']}={actual[mismatch:mismatch + 5]}"
            f"{distribution_detail}"
        )
    print(
        f"[compresslm-parity] PASS: {reference['backend']} and "
        f"{candidate['backend']} greedy outputs match for {len(expected)} tokens"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["hf", "vllm", "compare"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--model-dir")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--prompt-length", type=int, default=510)
    parser.add_argument("--prompt-text", default="", help="use this problem statement (raw SFT prompt format) instead of the synthetic repeated prompt")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--top-logprobs", type=int, default=0)
    parser.add_argument("--debug-hidden-dir", default="")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--no-cudagraph", action="store_true")
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--retain-history-layers", default="")
    parser.add_argument("--replace-codes-layers", default="")
    parser.add_argument("--packed-kv", action="store_true")
    parser.add_argument("--packed-kv-schedule", default="")
    parser.add_argument("--packed-kv-budget", type=int)
    parser.add_argument("--layer-quotas", dest="layer_quotas_path", default="", help="per-layer quota schedule json for masked vLLM serving of a per-layer trained schedule "
                                                           "(export keeps max-quota code rows; layers with smaller quotas hide the extra slots). Also the HF reference quotas.")
    parser.add_argument("--hf-quotas", default="", help="quota schedule json for the HF reference generator only (plain, non-packed vLLM serving of a "
                                                        "uniform-quota checkpoint exported with --code-rows)")
    parser.add_argument("--result-label", default="")
    parser.add_argument("--reference-result", default="")
    parser.add_argument("--candidate-result", default="")
    args = parser.parse_args()
    args.retain_history_layers = parse_layer_spec(
        args.retain_history_layers
    )
    args.replace_codes_layers = parse_layer_spec(
        args.replace_codes_layers
    )
    args.packed_kv_quotas = load_packed_kv_schedule(
        args.packed_kv_schedule,
        args.packed_kv_budget,
    )
    args.packed_kv = bool(args.packed_kv or args.packed_kv_quotas)
    args.hf_quotas = load_packed_kv_schedule(args.hf_quotas, None) if args.hf_quotas else ()
    args.layer_quotas = ()
    if args.layer_quotas_path:
        from compresslm_schedules import load_layer_quotas, load_quota_rows
        args.layer_quotas = load_layer_quotas(args.layer_quotas_path)
        if not args.hf_quotas:
            args.hf_quotas = load_quota_rows(args.layer_quotas_path)
        if args.packed_kv:
            parser.error("--layer-quotas is for masked (non-packed) serving; do not combine with --packed-kv")
    if args.hf_quotas and args.packed_kv:
        parser.error("--hf-quotas is for plain (non-packed) serving; do not combine with --packed-kv")
    if args.packed_kv and args.retain_history_layers:
        parser.error("packed KV and retained-history parity are mutually exclusive")
    if args.eager and args.no_cudagraph:
        parser.error("--eager and --no-cudagraph are mutually exclusive")
    if not set(args.replace_codes_layers).issubset(
        args.retain_history_layers
    ):
        parser.error(
            "--replace-codes-layers must be a subset of "
            "--retain-history-layers"
        )
    if (
        args.backend == "vllm"
        and args.retain_history_layers
        and args.attention_backend.replace("_", "").upper() != "FLASHINFER"
    ):
        parser.error("selective history parity requires FLASHINFER")

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    if args.backend == "compare":
        compare(prefix, args.reference_result, args.candidate_result)
        return
    if not args.checkpoint or not args.model_dir:
        parser.error("--checkpoint and --model-dir are required for decoding")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, local_files_only=True
    )
    if args.prompt_text:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from prompt_format import format_user_content
        prompt = tokenizer.encode(format_user_content(args.prompt_text, False), add_special_tokens=True)
    else:
        prompt = make_prompt(tokenizer, args.prompt_length)
    output, top_logprobs = (
        run_hf(args, prompt)
        if args.backend == "hf"
        else run_vllm(args, prompt)
    )
    write_result(
        prefix.with_suffix(f".{args.backend}.json"),
        args.result_label or args.backend,
        prompt,
        output,
        top_logprobs,
    )


if __name__ == "__main__":
    main()
