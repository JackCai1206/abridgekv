"""Evaluate Qwen SFT checkpoints on local MATH-500 with faithful CompressLM vLLM."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_math500 import grade as legacy_grade
from mv_grade import mv_grade_text, use_mv
from prompt_format import boxed_instruction_enabled, format_user_content

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")


def extract_last_boxed(text: str) -> str | None:
    """Extract the last balanced ``\boxed{...}``, including nested LaTeX."""
    marker = r"\boxed{"
    result = None
    search_from = 0
    while True:
        start = text.find(marker, search_from)
        if start < 0:
            return result
        content_start = start + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] in "{}" and (
                index == 0 or text[index - 1] != "\\"
            ):
                depth += 1 if text[index] == "{" else -1
            index += 1
        if depth == 0:
            result = text[content_start:index - 1].strip()
            search_from = index
        else:
            search_from = content_start


def strict_grade(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    semantic = (
        mv_grade_text(rf"\boxed{{{pred}}}", gold, kind="math")
        if use_mv()
        else None
    )
    return bool(semantic) if semantic is not None else legacy_grade(pred, gold)


def token_sha256(token_ids) -> str:
    return hashlib.sha256(
        ",".join(map(str, token_ids)).encode("ascii")
    ).hexdigest()


def load_rows(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def parse_indices(text: str, size: int) -> list[int]:
    if not text:
        return list(range(size))
    try:
        indices = [
            int(value) for value in re.split(r"[,:\s]+", text) if value
        ]
    except ValueError as error:
        raise ValueError("problem indices must be separated integers") from error
    if not indices or len(indices) != len(set(indices)):
        raise ValueError("problem indices must be non-empty and unique")
    if min(indices) < 0 or max(indices) >= size:
        raise ValueError(f"problem indices must be in [0, {size - 1}]")
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/math_eval/math500.jsonl"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--problem-indices", default="")
    parser.add_argument("--num-problems", type=int, default=0)
    parser.add_argument("--expected-num-problems", type=int, default=0)
    parser.add_argument("--prompt-repeats", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=0)
    parser.add_argument("--num-generations", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=24576)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use-compresslm", action="store_true")
    parser.add_argument("--compresslm-eager", action="store_true")
    parser.add_argument("--compresslm-no-cudagraph", action="store_true")
    parser.add_argument(
        "--compresslm-packed-kv",
        action="store_true",
        help="Pack completed code KVs by physical layer/head quota.",
    )
    parser.add_argument(
        "--compresslm-packed-kv-schedule",
        default="",
        help="2D quota JSON, or an object containing 'quotas'/'budgets'.",
    )
    parser.add_argument(
        "--compresslm-layer-quotas",
        default="",
        help="Per-layer quota schedule JSON (layer-uniform rows) for masked serving on the stock cache.",
    )
    parser.add_argument(
        "--compresslm-packed-kv-budget",
        type=int,
        help="Average-K key when loading a quota-analysis JSON file.",
    )
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--save-completions", action="store_true")
    parser.add_argument(
        "--chat-template-prompt",
        action="store_true",
        help="Wrap the user content in the tokenizer chat template (for untrained base models) instead of the raw SFT prompt path.",
    )
    args = parser.parse_args()
    packed_kv_quotas = ()
    if args.compresslm_packed_kv_schedule:
        packed_payload = json.loads(
            Path(args.compresslm_packed_kv_schedule).read_text()
        )
        if isinstance(packed_payload, dict):
            direct = packed_payload.get(
                "packed_kv_quotas", packed_payload.get("quotas")
            )
            if direct is not None:
                packed_payload = direct
            elif "budgets" in packed_payload:
                if args.compresslm_packed_kv_budget is None:
                    parser.error(
                        "--compresslm-packed-kv-budget is required for a "
                        "quota-analysis JSON"
                    )
                entry = packed_payload["budgets"].get(
                    str(args.compresslm_packed_kv_budget)
                )
                if not isinstance(entry, dict) or "train_K_lh" not in entry:
                    parser.error(
                        f"packed KV budget {args.compresslm_packed_kv_budget} is absent"
                    )
                packed_payload = entry["train_K_lh"]
        if not isinstance(packed_payload, list) or not packed_payload:
            parser.error(
                "packed KV schedule must be a non-empty 2D JSON array or "
                "an object containing 'packed_kv_quotas'/'quotas'/'budgets'"
            )
        packed_kv_quotas = tuple(
            tuple(int(v) for v in row) for row in packed_payload
        )
        args.compresslm_packed_kv = True
    if args.compresslm_packed_kv and not args.use_compresslm:
        parser.error("packed KV requires --use-compresslm")
    if args.compresslm_eager and args.compresslm_no_cudagraph:
        parser.error("CompressLM eager and no-cudagraph modes conflict")
    if args.problem_indices and args.num_problems:
        parser.error("--problem-indices conflicts with --num-problems")
    if args.prompt_repeats < 1:
        parser.error("--prompt-repeats must be positive")
    if args.prompt_batch_size < 0:
        parser.error("--prompt-batch-size must be non-negative")

    all_rows = load_rows(args.data)
    indices = parse_indices(args.problem_indices, len(all_rows))
    if args.num_problems:
        indices = indices[:args.num_problems]
    indices = [
        index
        for index in indices
        for _ in range(args.prompt_repeats)
    ]
    if (
        args.expected_num_problems
        and len(indices) != args.expected_num_problems
    ):
        raise ValueError(
            f"expected {args.expected_num_problems} problems, got "
            f"{len(indices)}; check submission environment serialization"
        )
    rows = [all_rows[index] for index in indices]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    use_boxed_instruction = boxed_instruction_enabled()
    expected_identity = {
        "task": "MATH-500",
        "requested_problem_indices": indices,
        "prompt_repeats": args.prompt_repeats,
        "prompt_batch_size": args.prompt_batch_size,
        "num_generations": args.num_generations,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "use_compresslm": args.use_compresslm,
        "attention_backend": (
            args.attention_backend if args.use_compresslm else None
        ),
        "boxed_instruction": use_boxed_instruction,
        "chat_template_prompt": bool(args.chat_template_prompt),
        "grading": "strict-last-boxed-math-verify",
        "completion_texts_omitted": not args.save_completions,
    }
    existing = None
    if args.output.is_file():
        existing = json.loads(args.output.read_text())
        actual_identity = {
            key: existing.get(key) for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                f"existing output identity mismatch: "
                f"{actual_identity} != {expected_identity}"
            )
        completed_indices = existing.get("problem_indices", [])
        if completed_indices != indices[:len(completed_indices)]:
            raise ValueError("existing output is not a prefix of requested indices")
        completed_problems = existing.get("per_problem", [])
        if (
            len(completed_problems) != len(completed_indices)
            or [
                problem.get("dataset_index")
                for problem in completed_problems
            ] != completed_indices
            or any(
                len(problem.get("samples", [])) != args.num_generations
                for problem in completed_problems
            )
        ):
            raise ValueError("existing output contains incomplete prompt cohorts")
        if len(completed_indices) == len(indices):
            print(f"[math500] complete output already exists: {args.output}")
            return

    from vllm import LLM, SamplingParams

    llm_kwargs = {
        "model": str(args.model_dir),
        "dtype": "bfloat16",
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        # GPU_MEM_UTIL: the packed CompressLM sidecar allocates its pages OUTSIDE vLLM's accounting
        # (~19 MB per completed block per sequence), so packed runs must leave the engine less of the card.
        "gpu_memory_utilization": float(os.environ.get("GPU_MEM_UTIL", "0.90")),
        "enable_prefix_caching": False,
    }
    compress_config = None
    if args.use_compresslm:
        from vllm.config.compresslm import CompressLMConfig

        metadata_path = args.model_dir / "compresslm_config.json"
        metadata = json.loads(metadata_path.read_text())
        metadata.pop("code_token_ids", None)
        # the export also records provenance (code_rows_of_k_max, ...) that is not a config field
        metadata = {k: v for k, v in metadata.items() if k in CompressLMConfig.__dataclass_fields__ and not k.startswith("_")}
        metadata["packed_kv"] = args.compresslm_packed_kv
        metadata["packed_kv_k_max"] = metadata["num_code_tokens"]
        metadata["packed_kv_quotas"] = packed_kv_quotas
        if args.compresslm_layer_quotas:
            from compresslm_schedules import load_layer_quotas
            metadata["layer_code_quotas"] = load_layer_quotas(args.compresslm_layer_quotas)
            metadata["physical_layer_quotas"] = os.environ.get("COMPRESSLM_PHYSICAL_QUOTAS") == "1"
        compress_config = CompressLMConfig(**metadata)
        physical_quotas = compress_config.uses_physical_layer_quotas
        if (
            compress_config.uses_selective_history
            or compress_config.uses_layer_quotas
            or (compress_config.code_excluded_tail and not compress_config.uses_packed_kv)
        ) and (args.attention_backend or "").replace("_", "").upper() != "FLASHINFER":
            print(f"[math500] CompressLM custom masks need FLASHINFER; overriding attention backend {args.attention_backend}")
            args.attention_backend = "FLASHINFER"
        llm_kwargs.update({
            "compresslm_config": compress_config,
            "enforce_eager": args.compresslm_eager or compress_config.uses_packed_kv
            or (compress_config.uses_layer_quotas and not physical_quotas and os.environ.get("COMPRESSLM_LAYER_PIECEWISE") != "1"),
            "attention_backend": args.attention_backend,
        })
        if args.compresslm_no_cudagraph or (compress_config.uses_layer_quotas and not physical_quotas):
            llm_kwargs["compilation_config"] = {"cudagraph_mode": "NONE"}
        if compress_config.uses_layer_quotas and not physical_quotas and os.environ.get("COMPRESSLM_LAYER_PIECEWISE") == "1":
            llm_kwargs["compilation_config"] = {"cudagraph_mode": "PIECEWISE"}   # attention eager (custom masks), the rest in graphs

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    # Exact prep_math_sft.py prompt path: raw format_user_content followed by
    # tokenizer(..., add_special_tokens=True), not the tokenizer chat template.
    if args.chat_template_prompt:
        prompts = [
            {
                "prompt_token_ids": tokenizer.apply_chat_template(
                    [{"role": "user", "content": format_user_content(row["problem"], use_boxed_instruction)}],
                    tokenize=True,
                    add_generation_prompt=True,
                )
            }
            for row in rows
        ]
    else:
        prompts = [
            {
                "prompt_token_ids": tokenizer.encode(
                    format_user_content(row["problem"], use_boxed_instruction),
                    add_special_tokens=True,
                )
            }
            for row in rows
        ]
    if compress_config:
        max_prompt = max(len(prompt["prompt_token_ids"]) for prompt in prompts)
        max_content = max_prompt + args.max_tokens
        transitions = (max_content - 1) // compress_config.block_size
        max_logical = max_content + transitions * (
            compress_config.num_code_tokens + compress_config.replay_tail
        )
        print(
            f"[math500] CompressLM context: max_prompt={max_prompt} "
            f"max_logical={max_logical} max_model_len={args.max_model_len}",
            flush=True,
        )
        if max_logical >= args.max_model_len:
            raise ValueError(
                f"max_model_len={args.max_model_len} must exceed {max_logical}"
            )

    params = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,
        bad_words=compress_config.code_tokens if compress_config else None,
    )
    prompt_batch_size = args.prompt_batch_size or len(prompts)
    per_problem = list(existing.get("per_problem", [])) if existing else []
    generation_seconds = (
        float(existing.get("generation_seconds", 0.0)) if existing else 0.0
    )

    def build_result() -> dict:
        samples = [
            sample
            for problem in per_problem
            for sample in problem["samples"]
        ]
        total = len(samples)
        total_tokens = sum(int(sample["gen_tokens"]) for sample in samples)
        correct = sum(bool(sample["correct"]) for sample in samples)
        semantic_correct = sum(
            bool(sample["correct_semantic_full_text"]) for sample in samples
        )
        no_answer = sum(
            sample.get("pred_last_boxed") is None for sample in samples
        )
        truncated = sum(
            sample.get("stop_reason") == "length" for sample in samples
        )
        return {
            **expected_identity,
            "num_problems": len(per_problem),
            "problem_indices": [
                int(problem["dataset_index"]) for problem in per_problem
            ],
            "total_generations": total,
            "accuracy": correct / max(total, 1),
            "semantic_full_text_accuracy": semantic_correct / max(total, 1),
            "no_answer_rate": no_answer / max(total, 1),
            "truncated_rate": truncated / max(total, 1),
            "mean_generation_tokens": total_tokens / max(total, 1),
            "total_visible_tokens": total_tokens,
            "generation_seconds": generation_seconds,
            "visible_tokens_per_second": (
                total_tokens / max(generation_seconds, 1e-9)
            ),
            "per_problem": per_problem,
        }

    for batch_start in range(
        len(per_problem), len(prompts), prompt_batch_size
    ):
        batch_end = min(batch_start + prompt_batch_size, len(prompts))
        if len(prompts) > prompt_batch_size:
            print(
                f"[math500] Prompt cohort "
                f"[{batch_start}, {batch_end})/{len(prompts)}",
                flush=True,
            )
        generation_start = time.perf_counter()
        outputs = llm.generate(prompts[batch_start:batch_end], params)
        generation_seconds += time.perf_counter() - generation_start
        if len(outputs) != batch_end - batch_start:
            raise RuntimeError(
                f"prompt cohort [{batch_start}, {batch_end}) returned "
                f"{len(outputs)} outputs"
            )
        for dataset_index, row, output in zip(
            indices[batch_start:batch_end],
            rows[batch_start:batch_end],
            outputs,
        ):
            if len(output.outputs) != args.num_generations:
                raise RuntimeError(
                    f"MATH-500 index {dataset_index} returned "
                    f"{len(output.outputs)} samples; expected "
                    f"{args.num_generations}"
                )
            samples = []
            for completion in output.outputs:
                text = completion.text
                pred = extract_last_boxed(text)
                ok = strict_grade(pred, row["answer"])
                semantic = (
                    mv_grade_text(text, row["answer"], kind="math")
                    if use_mv()
                    else None
                )
                sample = {
                    "pred_last_boxed": pred,
                    "correct": ok,
                    "correct_semantic_full_text": ok or bool(semantic),
                    "gen_tokens": len(completion.token_ids),
                    "stop_reason": getattr(completion, "finish_reason", None),
                    "token_sha256": token_sha256(completion.token_ids),
                }
                if args.save_completions:
                    sample["text"] = text
                samples.append(sample)
            per_problem.append({
                "dataset_index": dataset_index,
                "subject": row.get("subject"),
                "level": row.get("level"),
                "gold": row["answer"],
                "samples": samples,
            })
        atomic_write_json(args.output, build_result())
        print(
            f"[math500] checkpointed "
            f"{len(per_problem)}/{len(prompts)} problems",
            flush=True,
        )

    result = build_result()
    print(
        f"[math500][RESULT] n={len(per_problem)} "
        f"accuracy={result['accuracy']:.4f} "
        f"semantic={result['semantic_full_text_accuracy']:.4f} "
        f"no_answer={result['no_answer_rate']:.4f} "
        f"truncated={result['truncated_rate']:.4f} "
        f"mean_tokens={result['mean_generation_tokens']:.1f} "
        f"tok/s={result['visible_tokens_per_second']:.2f}",
        flush=True,
    )
    print(f"[math500] wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
    # vLLM's engine-core shutdown occasionally hangs at interpreter teardown (AIME shard 2 of job 2149 sat for 5.5 h after
    # writing its results and the whole eval hit its time limit). Results are on disk: leave without tearing down.
    # The engine core is a child process; a hard exit of this process alone leaves a hung core orphaned and still holding the GPU
    # (uniform suite job 2152: an orphan kept 255 GB on one GPU and every later shard on it failed to start). Kill children first.
    sys.stdout.flush(); sys.stderr.flush()
    try:
        import psutil
        for c in psutil.Process().children(recursive=True):
            try: c.kill()
            except Exception: pass
    except Exception:
        os.system(f"pkill -KILL -P {os.getpid()}")
    os._exit(0)
