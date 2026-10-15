# Data

Nothing large is stored in the repository. `bash launch/prep_data.sh` rebuilds everything from the Hugging Face Hub.

## Training pool

`scripts/prep_math_sft.py` draws 400,000 prompt/response pairs from `nvidia/OpenMathReasoning` (split `cot`),
keeps responses with a boxed answer that end in EOS, tokenizes them with the Qwen3-4B-Base tokenizer, and buckets
them by total length between 512 and 16,384 tokens (60% up to 4K, 25% up to 8K, 15% up to 16K), holding out 64
sequences per band for held-out cross-entropy. The result (about 1.93B response tokens) is written to
`$POOL`, by default `outputs/data/math_sft_long_qwen3_400000.pkl`. Training reads it in a fixed order that does not
depend on the number of GPUs (`DATA_STREAM=canonical:8x1x32` in `launch/train.sh`).

`CODE=1 bash launch/prep_data.sh` also builds the 120,000-pair coding pool from `nvidia/OpenCodeReasoning`
(`scripts/prep_code_sft.py`) used for the prototyping-scale coding results.

## Evaluation suite

`data/math_eval/SUITE.json` lists the 446-problem suite: AIME 2022-2026, AMC 2022-23, HMMT February 2023-2025,
BRUMO 2025, CMIMC 2025 and SMT 2025, with each set's source and era (before 2025, or after the base model's knowledge cutoff).
`scripts/build_math_eval_sets.py` and `scripts/dl_math_eval.py` write one JSONL file per set (plus MATH-500) into
this directory.

69 suite problems (68 from AIME 2022-2023 and AMC 2022-23, one borderline BRUMO problem) share at least half of their
word 13-grams with a single training prompt; `results/contamination_math_sft_400k.json` lists them
(`scripts/check_suite_contamination.py`). The paper repeats its comparisons without these problems.
