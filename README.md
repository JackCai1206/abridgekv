# AbridgeKV

Code for **AbridgeKV: Compressing Long Reasoning into the KV Cache of Learned Pause Tokens**
(Jack Cai, Lijie Yang, Vasilis Kontonis, Dimitris Papailiopoulos, Shivam Garg).

AbridgeKV trains a model to compress its own output. After every block of C = 512 generated tokens, the model reads
K learned pause tokens: extra input tokens with trained embeddings that produce no output. Once the pause tokens
have encoded the block, the block's KV entries are freed and later tokens attend to the pause tokens' entries
instead. Three choices make this work:

- the pause tokens' entries take positions **inside the block they replace**, so compressing a block adds no
  positions and the model needs no extra position range;
- the cache budget is **allocated per layer** according to the measured value of each layer's entries (6 layers keep
  256 rows per block, 18 keep 128, 12 keep 64; mean 128, a quarter of the generated cache);
- compression operates on **large blocks** (512 tokens).

## Results

Qwen3-4B-Base, fine-tuned on 2B response tokens of `nvidia/OpenMathReasoning`; every arm uses the same data and
recipe. Suite: 446 problems from AIME, HMMT, AMC, BRUMO, CMIMC and SMT, 8 samples per problem, 32K-token cap.

| Arm | KV rows per 512 tokens | Suite avg@8 [95% CI] | Δ vs. full KV (paired) | samples reaching the cap |
|---|---:|---:|---:|---:|
| Full-KV SFT | 512 | 43.8 [39.8, 47.6] | — | 3.4% |
| **AbridgeKV (per-layer)** | 128 | **43.0** [39.1, 46.9] | −0.8 [−2.3, +0.8] | 3.2% |
| uniform allocation | 128 | 41.5 [37.6, 45.3] | −2.3 [−3.8, −0.9] | 3.4% |
| appended positions | 128 | 35.3 [31.6, 39.0] | −8.5 [−10.3, −6.8] | 3.6% |
| LightThinker, our stack | 97 | 33.3 [29.6, 37.0] | −10.5 [−12.5, −8.7] | 5.4% |
| R-KV, budget 4096 | 144 | 42.6 [38.5, 46.6] | −1.2 [−2.7, +0.3] | 31.7% |
| R-KV, budget 2048 | 74 | 35.6 [31.7, 39.5] | −8.2 [−10.2, −6.3] | 55.8% |

Served in vLLM, the per-layer model's cache is 3.5× smaller per sequence, and a saturated H100 finishes a problem
3.2× faster than with full KV.

## Layout

| Path | Contents |
|---|---|
| `abridgekv/` | model wrapper and training: `compress_forward.py` (pause tokens, masks, in-block positions), `sft_ddp.py` (trainer), `data_stream.py` (fixed data order), `gen_compress.py` (reference generation), `lightthinker.py` (baseline) |
| `launch/` | one script per step of the pipeline below; shared settings in `launch/env.sh` |
| `scripts/` | data preparation, schedule measurement, export, evaluation, benchmarks, parity checks and tests |
| `third_party/vllm_overlay/` | vLLM 0.13.0 overlay that serves the models ([README](third_party/vllm_overlay/README.md)) |
| `results/schedules/` | the measured per-layer schedule used in the paper (math, and the coding-domain re-measurement) |
| `paper/` | the evaluation records behind the paper's tables and figures, and the scripts that draw them |
| `data/` | evaluation suite definition ([README](data/README.md)); everything else is downloaded |

## Setup

Two environments: training needs recent `transformers`; serving needs vLLM 0.13.0, which pins older versions.

```bash
# training, probes, export
pip install -r requirements.txt
# serving and evaluation (separate environment)
pip install vllm==0.13.0 math-verify==0.9.0 datasets
bash third_party/vllm_overlay/install_overlay.sh
```

Outputs go to `$OUTPUT_DIR` (default `./outputs`). The paper's runs used one node of 8× H100-80GB per training run
(about 15 hours each) and one H100 or A100-80GB per evaluation shard.

## Reproducing the paper

```bash
bash launch/prep_data.sh                                  # training pool + evaluation sets

ARM=uniform  bash launch/train.sh                         # 128 code rows per block in every layer
RUN_DIR=outputs/runs/uniform_c512_k128 bash launch/measure_schedule.sh   # -> perlayer_schedule.json
ARM=perlayer bash launch/train.sh                         # AbridgeKV; SCHEDULE=<json> to use your own schedule
ARM=vanilla  bash launch/train.sh                         # full-KV reference
ARM=appended bash launch/train.sh                         # ablation: code rows at appended positions
bash launch/train_lightthinker.sh                         # LightThinker baseline

RUN_DIR=outputs/runs/perlayer_c512_k128 ARM=perlayer bash launch/export.sh   # -> $RUN_DIR/hf
MODEL=outputs/runs/perlayer_c512_k128/hf ARM=perlayer TAG=perlayer bash launch/eval_suite.sh
python3 scripts/summarize_suite.py --root outputs/eval --out outputs/suite.json --md outputs/suite.md
```

`launch/train.sh` resumes from the newest complete checkpoint when re-run. The data order is fixed and independent
of the GPU count, so runs on other hardware see the same examples at every step. The measured schedule is
checked in (`results/schedules/sched_trunc_perlayer_mean128.json`), so the per-layer arm can be trained without
the uniform run first.

**R-KV.** Install the authors' vLLM port ([Zefan-Cai/R-KV](https://github.com/Zefan-Cai/R-KV)) in its own
environment and run `ARM=rkv RKV_BUDGET=4096 MODEL=<full-KV export> bash launch/eval_suite.sh` there. The script
sizes the port's scoring buffer from the budget; with a smaller buffer the port silently keeps the full cache.

**Speed.** `bash third_party/install_shadow.sh async $VLLM_SHADOW` builds a patched copy of vLLM with asynchronous
scheduling, then `MODEL=<export> ARM=perlayer TAG=perlayer VLLM_SHADOW=$VLLM_SHADOW bash launch/bench.sh` replays 512
requests at the arm's suite generation lengths and records the time to finish.

**Tables and figures.** `cd paper && python3 build_tables.py && python3 plots.py` regenerates every table and
figure from the records in `paper/results.json` and `paper/data/`.

## Tests

```bash
python3 scripts/test_compresslm_vllm_state.py      # vLLM scheduler state, CPU
python3 scripts/test_compresslm_layer_quotas.py    # per-layer quota masks and spans, CPU
python3 scripts/test_compresslm_physical_quotas.py # one KV group per quota, CPU
python3 scripts/test_sft_head_code_quotas.py       # trainer quota handling, CPU
python3 scripts/test_math_grading.py               # answer grading
LIGHTTHINKER_REPO=<clone of zjunlp/LightThinker> python3 scripts/test_lightthinker_mask.py
LT_SHADOW=<dir from install_shadow.sh lt> python3 scripts/test_lightthinker_vllm_state.py
```

On a GPU, `scripts/check_compresslm_vllm_parity.py` and `scripts/check_teacher_forced_parity.py` compare vLLM's
token log-probabilities with the training-time forward pass, and `scripts/check_full_attention_control.py` checks
that a served model attending to the evicted text would be detected.

## Citation

```bibtex
@article{cai2026abridgekv,
  title  = {AbridgeKV: Compressing Long Reasoning into the KV Cache of Learned Pause Tokens},
  author = {Cai, Jack and Yang, Lijie and Kontonis, Vasilis and Papailiopoulos, Dimitris and Garg, Shivam},
  year   = {2026}
}
```

## License

Apache-2.0. The vLLM overlay derives from vLLM (Apache-2.0) and the microsoft/memento overlay (MIT); see `NOTICE`.
