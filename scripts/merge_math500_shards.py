"""Merge sharded MATH-500 records and recompute aggregate metrics."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


def parse_ints(value: str) -> list[int]:
    parts = [int(part) for part in re.split(r"[,:\s]+", value) if part]
    if len(parts) == 2 and ":" in value:
        return list(range(parts[0], parts[1] + 1))
    return parts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--arm", default="compressed")
    parser.add_argument("--expected-indices", default="0:499")
    args = parser.parse_args()

    expected_indices = parse_ints(args.expected_indices)
    records: list[dict[str, Any]] = []
    by_index: dict[int, dict[str, Any]] = {}
    config_keys = (
        "task",
        "num_generations",
        "seed",
        "max_tokens",
        "temperature",
        "use_compresslm",
        "attention_backend",
        "boxed_instruction",
        "grading",
        "prompt_batch_size",
    )
    expected_config = None
    for shard_dir in args.shard_dir:
        path = shard_dir / f"{args.arm}.json"
        record = json.loads(path.read_text())
        config = {key: record.get(key) for key in config_keys}
        if expected_config is None:
            expected_config = config
        elif config != expected_config:
            raise ValueError(
                f"incompatible shard configuration in {path}: "
                f"{config} != {expected_config}"
            )
        records.append(record)
        for problem in record["per_problem"]:
            index = int(problem["dataset_index"])
            if index in by_index:
                raise ValueError(f"duplicate MATH-500 dataset index {index}")
            by_index[index] = problem

    actual_indices = sorted(by_index)
    if actual_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(actual_indices))
        extra = sorted(set(actual_indices) - set(expected_indices))
        raise ValueError(
            f"MATH-500 index coverage mismatch: missing={missing}, extra={extra}"
        )

    problems = [by_index[index] for index in actual_indices]
    samples = [
        sample
        for problem in problems
        for sample in problem["samples"]
    ]
    expected_samples = int(records[0]["num_generations"])
    if {len(problem["samples"]) for problem in problems} != {expected_samples}:
        raise ValueError("inconsistent samples per problem")

    total = len(samples)
    total_tokens = sum(int(sample["gen_tokens"]) for sample in samples)
    generation_seconds = sum(
        float(record["generation_seconds"]) for record in records
    )
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

    merged = dict(records[0])
    merged.update(
        {
            "num_problems": len(problems),
            "problem_indices": actual_indices,
            "prompt_repeats": 1,
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
            "completion_texts_omitted": any(
                record.get("completion_texts_omitted", True)
                for record in records
            ),
            "per_problem": problems,
            "source_shards": [str(path) for path in args.shard_dir],
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.arm}.json"
    output.write_text(json.dumps(merged, indent=2) + "\n")
    print(
        f"[merge-math500] n={len(problems)} "
        f"accuracy={merged['accuracy']:.4f} "
        f"no_answer={merged['no_answer_rate']:.4f} "
        f"truncated={merged['truncated_rate']:.4f} -> {output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
