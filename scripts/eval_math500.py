"""MATH-500 evaluation helpers (answer extraction and grading) used by eval_math500_compresslm.py, plus a
standalone evaluator for models served with the block-masking overlay inherited from microsoft/memento.

Uses the vLLM overlay in third_party/vllm_overlay (install with install_overlay.sh)
to provide BlockMaskingConfig at inference time, matching the paper's setup.

Usage:
    python eval_math500.py --model-dir /path/to/hf_export
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root (portable)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # scripts/ for mv_grade
from mv_grade import mv_grade_text, use_mv  # noqa: E402
from prompt_format import format_user_content, boxed_instruction_enabled  # noqa: E402
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")


BOXED_RE = re.compile(r"\\boxed\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}")


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last \\boxed{...} expression as the model's answer."""
    matches = BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def normalize_for_compare(s: object) -> str:
    s = str(s)
    s = s.replace(" ", "").replace("\\", "").replace("$", "")
    s = s.replace("{", "").replace("}", "")
    return s.lower()


def grade(pred: str | None, gold: object) -> bool:
    """Exact-string grade after normalization; sympy fallback for numeric/algebraic.
    For MATH-500 this is mostly fine because answers are short.
    """
    if pred is None:
        return False
    gold_text = str(gold)
    if normalize_for_compare(pred) == normalize_for_compare(gold_text):
        return True
    # sympy fallback
    try:
        from sympy import sympify, simplify
        a = sympify(pred.replace("\\", ""))
        b = sympify(gold_text.replace("\\", ""))
        return bool(simplify(a - b) == 0)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="HF-format model directory")
    ap.add_argument("--num-problems", type=int, default=500, help="how many MATH-500 problems")
    ap.add_argument("--num-generations", type=int, default=2, help="generations per problem")
    ap.add_argument("--max-tokens", type=int, default=30720)
    ap.add_argument("--max-model-len", type=int, default=32768,
                    help="vLLM context window. MUST be >= prompt+max_tokens (old hardcoded "
                         "16384 silently halved the 32K rollout and truncated long reasoning).")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--use-block-masking", action="store_true", help="enable block masking")
    ap.add_argument("--shard-idx", type=int, default=0, help="which shard of problems (0..num_shards-1)")
    ap.add_argument("--num-shards", type=int, default=1, help="split problems into N shards for parallel eval")
    ap.add_argument("--keep-last-n-blocks", type=int, default=0)
    ap.add_argument("--output", default="outputs/eval/math500_results.json")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading MATH-500 dataset")
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    if args.num_problems < len(ds):
        ds = ds.select(range(args.num_problems))
    # Apply sharding (deterministic slice).
    if args.num_shards > 1:
        N = len(ds)
        start = (args.shard_idx * N) // args.num_shards
        end = ((args.shard_idx + 1) * N) // args.num_shards
        ds = ds.select(range(start, end))
        print(f"  Shard {args.shard_idx}/{args.num_shards}: problems [{start}, {end}) — {len(ds)} problems")
    else:
        print(f"  {len(ds)} problems")

    # Build vLLM. If block masking is requested, pass BlockMaskingConfig.
    print(f"\nLoading model with vLLM from {args.model_dir}")
    from vllm import LLM, SamplingParams
    if args.max_tokens > args.max_model_len - 512:
        args.max_tokens = args.max_model_len - 512
        print(f"  [clamp] max_tokens -> {args.max_tokens} to fit max_model_len={args.max_model_len}")
    print(f"  max_model_len={args.max_model_len}, max_tokens={args.max_tokens}")
    llm_kwargs = dict(
        model=args.model_dir,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=False,  # required with block masking
    )
    if args.use_block_masking:
        try:
            from vllm.config.block_masking import BlockMaskingConfig
            bm = BlockMaskingConfig(
                enable=True,
                keep_last_n_blocks=args.keep_last_n_blocks,
                mask_delimiters=False,
                compact_on_summary_end=True,
                require_assistant_section=True,
                debug=os.environ.get("BM_DEBUG") == "1",  # per-block events flood a 500-problem run; opt-in
            )
            llm_kwargs["block_masking_config"] = bm
            print(f"  Block masking ENABLED: {bm}")
            print(f"    keep_last_n_blocks={args.keep_last_n_blocks}, mask_delimiters=False, "
                  f"compact_on_summary_end=True, debug=True")
        except ImportError as e:
            print(f"  ERROR: BlockMaskingConfig not available: {e}")
            raise  # don't silently fall back — user explicitly requested masking

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    # Build prompts (chat-template).
    def build_prompt(problem: str) -> str:
        # MATCH TRAINING via shared prompt_format; instruction OFF by default (current
        # matrix = raw problem). Set BOXED_INSTRUCTION=1 for instruction-trained ckpts.
        content = format_user_content(problem, boxed_instruction_enabled())
        msgs = [{"role": "user", "content": content}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    prompts = []
    golds = []
    for ex in ds:
        prompts.append(build_prompt(ex["problem"]))
        # MATH-500 calls the gold "answer"
        golds.append(ex["answer"])

    # Sampling: N generations per prompt.
    sp = SamplingParams(
        n=args.num_generations,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        skip_special_tokens=False,  # keep block markers
    )

    print(f"\nGenerating {args.num_generations} samples/problem ({args.num_problems} prompts)")
    outputs = llm.generate(prompts, sp)

    # Score: per-problem breakdown by category (right / wrong / no-answer)
    # + token usage and context length stats per generation.
    print(f"\nScoring")
    per_problem_acc = []
    results = []
    total_gens = 0
    n_correct = 0
    n_wrong_extracted = 0
    n_no_answer = 0
    gen_token_lens = []
    prompt_token_lens = []
    truncated_count = 0  # hit max_tokens (no stop token)
    for ex, out, gold in zip(ds, outputs, golds):
        prompt_len = len(out.prompt_token_ids)
        prompt_token_lens.append(prompt_len)
        correct_count = 0
        sample_records = []
        for completion in out.outputs:
            text = completion.text
            gen_len = len(completion.token_ids)
            gen_token_lens.append(gen_len)
            # vLLM reports stop reason; max_tokens hit = truncated
            stop_reason = getattr(completion, "finish_reason", None)
            if stop_reason == "length":
                truncated_count += 1
            pred = extract_boxed_answer(text)
            total_gens += 1
            # math_verify on the full text (recovers equiv answers); fall back to the
            # legacy extract+grade when unavailable/unparseable. USE_MATH_VERIFY=0 disables.
            mv_ok = mv_grade_text(text, gold, kind="math") if use_mv() else None
            if pred is None and mv_ok is not True:
                category = "no_answer"
                n_no_answer += 1
                ok = False
            else:
                ok = mv_ok if mv_ok is not None else grade(pred, gold)
                if ok:
                    category = "correct"
                    n_correct += 1
                else:
                    category = "wrong"
                    n_wrong_extracted += 1
            correct_count += int(ok)
            # Count block-format markers in generation (proxy for compaction count
            # since each summary_end triggers one compaction at inference).
            n_blocks = text.count("<|block_end|>")
            n_summaries_done = text.count("<|summary_end|>")
            sample_records.append({
                "text": text,
                "pred": pred,
                "correct": ok,
                "category": category,
                "gen_tokens": gen_len,
                "stop_reason": stop_reason,
                "n_block_end": n_blocks,
                "n_summary_end_inferred_compactions": n_summaries_done,
            })
        problem_acc = correct_count / max(len(out.outputs), 1)
        per_problem_acc.append(problem_acc)
        results.append({
            "problem": ex["problem"][:300],
            "gold": gold,
            "prompt_tokens": prompt_len,
            "samples": sample_records,
            "acc": problem_acc,
        })

    pass_at_1 = sum(per_problem_acc) / len(per_problem_acc) * 100
    pct_correct = 100 * n_correct / max(total_gens, 1)
    pct_wrong = 100 * n_wrong_extracted / max(total_gens, 1)
    pct_no_answer = 100 * n_no_answer / max(total_gens, 1)
    pct_truncated = 100 * truncated_count / max(total_gens, 1)

    def stats(xs):
        if not xs:
            return {}
        xs_sorted = sorted(xs)
        n = len(xs_sorted)
        return {
            "min": xs_sorted[0],
            "p25": xs_sorted[n // 4],
            "median": xs_sorted[n // 2],
            "p75": xs_sorted[3 * n // 4],
            "p90": xs_sorted[9 * n // 10],
            "p99": xs_sorted[min(n - 1, 99 * n // 100)],
            "max": xs_sorted[-1],
            "mean": sum(xs_sorted) / n,
        }

    gen_stats = stats(gen_token_lens)
    prompt_stats = stats(prompt_token_lens)
    # Block-format proxies: total block_ends, summary_ends, max blocks per gen.
    total_blocks = sum(s["n_block_end"] for r in results for s in r["samples"])
    total_compactions = sum(s["n_summary_end_inferred_compactions"] for r in results for s in r["samples"])
    blocks_per_gen = [s["n_block_end"] for r in results for s in r["samples"]]
    blocks_stats = stats(blocks_per_gen)

    print(f"\nMATH-500 results — {len(per_problem_acc)} problems, "
          f"{args.num_generations} gens each, {total_gens} total generations:")
    print(f"  pass@1 (per-problem avg correct): {pass_at_1:.2f}%")
    print(f"  ── per-generation breakdown ──")
    print(f"   right (boxed + matches gold):  {n_correct}/{total_gens}  = {pct_correct:.2f}%")
    print(f"   wrong (boxed but mismatched):  {n_wrong_extracted}/{total_gens}  = {pct_wrong:.2f}%")
    print(f"   no answer (no \\boxed{{}} found): {n_no_answer}/{total_gens}  = {pct_no_answer:.2f}%")
    print(f"   truncated (hit max_tokens):    {truncated_count}/{total_gens}  = {pct_truncated:.2f}%")
    print(f"  ── token / context stats ──")
    print(f"   prompt tokens:    {prompt_stats}")
    print(f"   gen tokens:       {gen_stats}")
    print(f"   blocks per gen:   {blocks_stats}")
    print(f"   total <|block_end|> emitted: {total_blocks}")
    print(f"   total <|summary_end|> (= compactions): {total_compactions}")

    with open(out_path, "w") as f:
        json.dump({
            "pass_at_1_pct": pass_at_1,
            "pct_correct": pct_correct,
            "pct_wrong": pct_wrong,
            "pct_no_answer": pct_no_answer,
            "pct_truncated": pct_truncated,
            "n_correct": n_correct,
            "n_wrong": n_wrong_extracted,
            "n_no_answer": n_no_answer,
            "n_truncated": truncated_count,
            "total_generations": total_gens,
            "num_problems": len(per_problem_acc),
            "num_generations": args.num_generations,
            "use_block_masking": args.use_block_masking,
            "prompt_token_stats": prompt_stats,
            "gen_token_stats": gen_stats,
            "blocks_per_gen_stats": blocks_stats,
            "total_block_ends": total_blocks,
            "total_summary_ends_inferred_compactions": total_compactions,
            "per_problem": results,
        }, f, indent=2)
    print(f"\nWrote detailed results to {out_path}")


if __name__ == "__main__":
    main()
