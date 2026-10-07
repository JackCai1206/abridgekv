#!/usr/bin/env python3
"""Controlled decode benchmark: physical KV memory, max concurrency and decode speed (docs/PLAN_efficiency_benchmark.md).

Drives vLLM's engine step loop in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0) so the paged-cache occupancy can be sampled
between steps, with every sequence forced to EXACTLY --len tokens (min_tokens == max_tokens, ignore_eos) so the compressed
and vanilla arms do identical work.

Modes
  --mode batch     : --batch B sequences of --len L; reports tok/s, per-step latency vs. context position, KV blocks in use.
  --mode saturate  : queue --requests N >> capacity with a large --max-num-seqs; reports the achieved max running sequences
                     (the concurrency a GPU sustains at this length) and the steady-state tok/s.
  --mode complete  : the capacity cost of a REAL workload. Replays --requests generation lengths drawn from the arm's own
                     32K suite outputs (--lengths-file, key --lengths-key; built by paper/make_workload.py), queues them all
                     at t=0 and runs every one to completion. Reports wall time per problem. Unlike "saturate", which stops
                     after a fixed step count while every sequence is still growing and so mostly times the start-up
                     phase in which no arm is memory-bound, this includes the long-context phase where the cache binds.

Arms are selected exactly like scripts/eval_aime.py: --model-dir is an HF export; --compresslm turns on the CompressLM
config from the export's compresslm_config.json; --layer-quotas <schedule.json> plus COMPRESSLM_PHYSICAL_QUOTAS=1 gives the
physical per-layer quota path; without it, COMPRESSLM_LAYER_PIECEWISE=1 gives the mask-based layered path.

  python3 scripts/bench_decode.py --model-dir <hf_export> --tag tpl_physical --compresslm \
      --layer-quotas results/schedules/sched_trunc_perlayer_mean128.json \
      --mode batch --batch 8 --len 4096 --out $OUTPUT_BASE/bench
"""
import argparse, json, os, sys, time
from pathlib import Path

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")     # in-process engine: lets us read the KV block pool
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def build_engine(args):
    from vllm import LLMEngine, EngineArgs
    kwargs = dict(
        model=args.model_dir, dtype="bfloat16", max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs, enable_prefix_caching=False,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.90")),
        disable_log_stats=True,
    )
    # VLLM_KV_CACHE_GIB: give the paged KV cache exactly this many GiB instead of what gpu_memory_utilization leaves over.
    # A fixed cache size, not a utilization fraction, so every arm gets the same bytes of KV whatever its activation and
    # CUDA-graph footprint (the fixed-KV sweep of the block-size trade-off, user 2026-09-24).
    if os.environ.get("VLLM_KV_CACHE_GIB"):
        kwargs["kv_cache_memory_bytes"] = int(float(os.environ["VLLM_KV_CACHE_GIB"]) * 2**30)
    compress_config = None
    if args.compresslm:
        from vllm.config.compresslm import CompressLMConfig
        meta = json.loads((Path(args.model_dir) / "compresslm_config.json").read_text())
        meta.pop("code_token_ids", None)
        meta = {k: v for k, v in meta.items() if k in CompressLMConfig.__dataclass_fields__ and not k.startswith("_")}
        if args.layer_quotas:
            from compresslm_schedules import load_layer_quotas
            meta["layer_code_quotas"] = load_layer_quotas(args.layer_quotas)
            meta["physical_layer_quotas"] = os.environ.get("COMPRESSLM_PHYSICAL_QUOTAS") == "1"
        compress_config = CompressLMConfig(**meta)
        physical = compress_config.uses_physical_layer_quotas
        kwargs["compresslm_config"] = compress_config
        kwargs["enforce_eager"] = bool(
            compress_config.uses_packed_kv
            or (compress_config.uses_layer_quotas and not physical and os.environ.get("COMPRESSLM_LAYER_PIECEWISE") != "1")
        )
        if compress_config.uses_layer_quotas and not physical:
            kwargs["compilation_config"] = (
                {"cudagraph_mode": "PIECEWISE"} if os.environ.get("COMPRESSLM_LAYER_PIECEWISE") == "1"
                else {"cudagraph_mode": "NONE"}
            )
        kwargs["attention_backend"] = args.attention_backend or "FLASHINFER"
    else:
        if args.attention_backend:
            kwargs["attention_backend"] = args.attention_backend
        if args.eager:
            kwargs["enforce_eager"] = True
        if args.cudagraph_mode:
            kwargs["compilation_config"] = {"cudagraph_mode": args.cudagraph_mode}
    # COMPRESSLM_CAPTURE_MAX: largest piecewise/FULL capture size (vLLM's default is 2 x max_num_seqs). Code steps carry
    # K rows per request (plus every other request's decode token when mixed), so they exceed the default and silently
    # run eager; raising it lets them run piecewise. Set it on BOTH sides of a comparison.
    if os.environ.get("COMPRESSLM_CAPTURE_MAX"):
        cc = dict(kwargs.get("compilation_config") or {})
        cc["max_cudagraph_capture_size"] = int(os.environ["COMPRESSLM_CAPTURE_MAX"])
        kwargs["compilation_config"] = cc
    # VLLM_ASYNC_SCHED=1: vLLM async scheduling (host prep of step t+1 overlaps the GPU running step t). CompressLM
    # supports it only in a package with the async scheduler edits (third_party/vllm_overlay_async); set it on BOTH sides.
    if os.environ.get("VLLM_ASYNC_SCHED") == "1":
        kwargs["async_scheduling"] = True
    engine = LLMEngine.from_engine_args(EngineArgs(**kwargs))
    return engine, compress_config


def kv_probe(engine):
    """(blocks_in_use, total_blocks, block_size, n_groups) of the paged cache, or Nones when it is not reachable."""
    try:
        sched = engine.engine_core.engine_core.scheduler
        pool = sched.kv_cache_manager.block_pool
        total = pool.num_gpu_blocks
        free = pool.get_num_free_blocks()
        groups = sched.kv_cache_manager.kv_cache_config.kv_cache_groups
        return total - free, total, groups[0].kv_cache_spec.block_size, len(groups)
    except Exception:
        return None, None, None, None


def kv_probe_groups(engine):
    """Blocks currently held per KV cache group, plus each group's code quota.

    This is the direct test that physical per-layer quotas really shrink the cache: with the S-PL-trunc split the
    six groups should hold blocks in proportion to their quotas, not equally. Returns (blocks_per_group, quotas)
    or (None, None) when the coordinator does not expose per-group managers.
    """
    try:
        sched = engine.engine_core.engine_core.scheduler
        mgrs = sched.kv_cache_manager.coordinator.single_type_managers
        per_group = [sum(len(b) for b in m.req_to_blocks.values()) for m in mgrs]
        groups = sched.kv_cache_manager.kv_cache_config.kv_cache_groups
        quotas = [getattr(g.kv_cache_spec, "code_quota", None) for g in groups]
        return per_group, quotas
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=os.environ.get("BENCH_OUT", "bench"))
    ap.add_argument("--compresslm", action="store_true")
    ap.add_argument("--layer-quotas", default=None)
    ap.add_argument("--attention-backend", default=None)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--cudagraph-mode", default=None)
    ap.add_argument("--mode", choices=["batch", "saturate", "complete"], default="batch")
    ap.add_argument("--lengths-file", default=None)
    ap.add_argument("--lengths-key", default=None)
    ap.add_argument("--len-scale", type=float, default=1.0)   # prototype only: shrink every length by this factor
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--requests", type=int, default=64)
    ap.add_argument("--len", dest="length", type=int, default=4096)
    ap.add_argument("--prompt-len", type=int, default=200)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--rep", type=int, default=0)
    ap.add_argument("--warmup-steps", type=int, default=8)
    # saturate mode: stop after one cohort's worth of steps. Peak concurrency is reached while the first admitted
    # sequences grow to full length; draining every queued request afterwards costs hours and adds no information.
    ap.add_argument("--max-steps", type=int, default=0)
    # --real-prompts (complete mode, opt-in; the LightThinker arm, 2026-09-26): prompt each request with the REAL problem
    # of its workload draw (eval prompt path: format_user_content + BOXED_INSTRUCTION, add_special_tokens) and sample
    # with --sample-temperature/--sample-top-p/--sample-top-k (per-request seed --sample-seed + i; -1 = no per-request
    # seed, which keeps vLLM on its batched sampler instead of per-request generators) instead of greedy on
    # the dummy prompt. For boundary-triggered compaction, which fires only when the model samples its splitter, the
    # dummy greedy prompt never compacts; the fixed-cadence arms do not depend on content and keep the default.
    ap.add_argument("--real-prompts", action="store_true")
    ap.add_argument("--sample-temperature", type=float, default=0.0)
    ap.add_argument("--sample-top-p", type=float, default=1.0)
    ap.add_argument("--sample-top-k", type=int, default=-1)
    ap.add_argument("--sample-seed", type=int, default=1234)
    args = ap.parse_args()

    engine, compress_config = build_engine(args)
    # Preemption census: when the paged cache runs out, vLLM V1 evicts a running request and later RECOMPUTES it from
    # token 0 (re-prefill, and for CompressLM a rebuild of every compacted block). Over-admitting arms can thrash here,
    # so count events and the tokens thrown away; they explain a slow step better than any throughput number.
    preempt = {"events": 0, "tokens_discarded": 0}
    try:
        _sched = engine.engine_core.engine_core.scheduler
        _orig_preempt = _sched._preempt_request
        def _counting_preempt(request, *a, **k):
            preempt["events"] += 1
            preempt["tokens_discarded"] += int(getattr(request, "num_computed_tokens", 0) or 0)
            return _orig_preempt(request, *a, **k)
        _sched._preempt_request = _counting_preempt
    except Exception as e:
        print(f"[bench] preemption census unavailable: {e}", flush=True)
    # COMPRESSLM_COUNT_BLOCKS=1: count finished (compacted) blocks, i.e. code steps completed per request. For boundary-
    # triggered arms (LightThinker) the compaction count depends on what the model samples, so it is the check that the
    # replay actually compresses at a realistic rate (visible tokens per compaction). Counting only; opt-in.
    blocks = {"finished": 0}
    if os.environ.get("COMPRESSLM_COUNT_BLOCKS") == "1":
        _p = getattr(engine.engine_core.engine_core.scheduler, "compresslm_processor", None)
        if _p is not None:
            _orig_fb = _p._finish_block
            def _counting_fb(*a, **k):
                blocks["finished"] += 1
                return _orig_fb(*a, **k)
            _p._finish_block = _counting_fb
        else:
            print("[bench] COMPRESSLM_COUNT_BLOCKS: no compresslm_processor", flush=True)
    # COMPRESSLM_STEP_PROFILE=1: record what every scheduler step contained (code step or not, #requests, #tokens),
    # so the run-to-completion wall time can be split into exclusive code steps vs decode steps.
    profile = os.environ.get("COMPRESSLM_STEP_PROFILE") == "1"
    sched_log = []
    if profile:
        _sch = engine.engine_core.engine_core.scheduler
        _orig_schedule = _sch.schedule
        _proc = getattr(_sch, "compresslm_processor", None)
        def _logging_schedule(*a, **k):
            # code-ready requests BEFORE scheduling (the burst a mixed step's row cap sees), and the requests that got
            # code rows (q>1; K>1 always); in a mixed step the rest of the scheduled requests are q=1 decodes.
            n_ready = sum(map(_proc.is_code_ready, _sch.running)) if _proc is not None else 0
            out = _orig_schedule(*a, **k)
            sched_log.append((bool(getattr(out, "compresslm_code_step", False)),
                              len(out.num_scheduled_tokens), int(out.total_num_scheduled_tokens),
                              n_ready, sum(1 for n in out.num_scheduled_tokens.values() if n > 1)))
            return out
        _sch.schedule = _logging_schedule
    # COMPRESSLM_STEP_TIMING=1: per engine step, split wall time into host phases (scheduler, state update, input prep,
    # attention-metadata build, sampling, bookkeeping incl. the sync that waits for the GPU) and the GPU span of the
    # model forward (CUDA events around _model_forward), plus the step's batch composition. Timing only: every wrapped
    # method is called unchanged. Written to <out>/<name>.steptiming.json; a summary goes into the main record.
    timing = os.environ.get("COMPRESSLM_STEP_TIMING") == "1"
    TKEYS = ("sched", "upd", "prep", "meta", "pre", "fwd", "samp", "book", "exec", "stok", "ufo")
    tcur, trows, tdefer = {}, [], []
    if timing:
        import torch
        _core = engine.engine_core.engine_core
        _mr = _core.model_executor.driver_worker.worker.model_runner
        def _twrap(obj, name, key, gpu=False, after=None):
            f = getattr(obj, name)
            def g(*a, **k):
                if gpu:
                    e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
                t = time.perf_counter()
                out = f(*a, **k)
                tcur[key] = tcur.get(key, 0.0) + time.perf_counter() - t
                if gpu:
                    e1.record(); tcur.setdefault("_ev", []).append((e0, e1))
                if after is not None:
                    after(out)
                return out
            setattr(obj, name, g)
        def _batch(out):
            n = list(out.num_scheduled_tokens.values())
            tcur["reqs"] = tcur.get("reqs", 0) + len(n)
            tcur["toks"] = tcur.get("toks", 0) + int(out.total_num_scheduled_tokens)
            tcur["code_rows"] = tcur.get("code_rows", 0) + sum(x for x in n if x > 1)
            tcur["code"] = tcur.get("code", 0) or int(bool(getattr(out, "compresslm_code_step", False)))
        for _n, _k in (("_update_states", "upd"), ("_prepare_inputs", "prep"), ("_build_attention_metadata", "meta"),
                       ("_preprocess", "pre"), ("_sample", "samp"), ("_bookkeeping_sync", "book"),
                       ("execute_model", "exec"), ("sample_tokens", "stok")):
            if hasattr(_mr, _n):
                _twrap(_mr, _n, _k)
        _twrap(_mr, "_model_forward", "fwd", gpu=True)
        _twrap(_core.scheduler, "update_from_output", "ufo")
        _twrap(_core.scheduler, "schedule", "sched", after=_batch)
    try:
        tok = engine.get_tokenizer()
    except Exception:
        tok = None
    from vllm import SamplingParams
    # Forced-length decoding so all arms do identical work; greedy so sampling cost is equal.
    # ignore_eos + max_tokens alone pins the length: CompressLM's stop rule counts VISIBLE tokens
    # (processor._check_stop: `visible_count >= request.max_tokens`) and rejects min_tokens != 0,
    # so min_tokens would only add an unsupported knob without changing the number of decoded tokens.
    sp = SamplingParams(
        temperature=0.0, max_tokens=args.length, ignore_eos=True,
        bad_words=(compress_config.code_tokens if compress_config else None),
    )
    n_req = args.batch if args.mode == "batch" else args.requests
    lengths = None
    if args.mode == "complete":
        W = json.loads(Path(args.lengths_file).read_text())
        lengths = [max(1, int(round(L * args.len_scale))) for L in W["arms"][args.lengths_key]["lengths"][:n_req]]
        assert len(lengths) == n_req, f"workload has {len(lengths)} lengths, asked for {n_req}"
        cap = args.max_model_len - args.prompt_len - 16
        assert max(lengths) <= cap or not compress_config, f"length {max(lengths)} exceeds max_model_len budget {cap}"
    prompt_ids = None
    if tok is not None:
        text = "Solve the following problem step by step. " * 64
        prompt_ids = tok(text).input_ids[: args.prompt_len]
    if not prompt_ids:
        prompt_ids = list(range(1000, 1000 + args.prompt_len))

    from vllm.inputs import TokensPrompt
    real = None
    if args.real_prompts:
        assert lengths is not None and tok is not None, "--real-prompts needs --mode complete and a tokenizer"
        from prompt_format import boxed_instruction_enabled, format_user_content
        root = Path(__file__).resolve().parents[1]
        rows = {}
        real = []
        for s_, idx_, _k in W["draws"][:n_req]:
            if s_ not in rows:
                rows[s_] = [json.loads(l) for l in (root / "data" / "math_eval" / f"{s_}.jsonl").read_text().splitlines() if l.strip()]
            real.append(tok.encode(format_user_content(rows[s_][idx_]["problem"], boxed_instruction_enabled()),
                                   add_special_tokens=True))
        print(f"[bench] real prompts: {len(real)} draws, mean prompt {sum(map(len, real)) / len(real):.0f} tok, "
              f"sampling T={args.sample_temperature} top_p={args.sample_top_p} top_k={args.sample_top_k} "
              f"boxed={boxed_instruction_enabled()}", flush=True)
    for i in range(n_req):
        if real is not None:
            sp_i = SamplingParams(
                temperature=args.sample_temperature, top_p=args.sample_top_p, top_k=args.sample_top_k,
                seed=(args.sample_seed + i) if args.sample_seed >= 0 else None, max_tokens=lengths[i], ignore_eos=True,
                bad_words=(compress_config.code_tokens if compress_config else None))
            engine.add_request(str(i), TokensPrompt(prompt_token_ids=list(real[i])), sp_i)
            continue
        sp_i = sp if lengths is None else SamplingParams(
            temperature=0.0, max_tokens=lengths[i], ignore_eos=True,
            bad_words=(compress_config.code_tokens if compress_config else None))
        engine.add_request(str(i), TokensPrompt(prompt_token_ids=list(prompt_ids)), sp_i)
    finish_t = {}      # complete mode: request id -> seconds from start to completion

    steps, finished = [], 0
    max_running, kv_peak = 0, 0
    total_blocks = block_size = n_groups = None
    group_blocks_peak, group_quotas = None, None   # sampled: per-group block census at the fullest sampled step
    t_start = time.perf_counter()
    t_steady = None
    tokens_at_steady = 0
    generated = 0
    truncated = False
    while engine.has_unfinished_requests():
        if args.max_steps and len(steps) >= args.max_steps:
            truncated = True
            break
        t0 = time.perf_counter()
        outs = engine.step()
        dt = time.perf_counter() - t0
        in_use, total_blocks, block_size, n_groups = kv_probe(engine)
        try:
            running = len(engine.engine_core.engine_core.scheduler.running)
        except Exception:
            running = None
        if in_use is not None:
            kv_peak = max(kv_peak, in_use)
        if running is not None:
            max_running = max(max_running, running)
        generated += sum(len(o.outputs[0].token_ids) if o.outputs else 0 for o in outs if o.finished)
        if len(steps) % 64 == 0:   # sampled, not per step: O(requests x groups) and we only need the peak
            gb, gq = kv_probe_groups(engine)
            if gb and (group_blocks_peak is None or sum(gb) > sum(group_blocks_peak)):
                group_blocks_peak, group_quotas = gb, gq
        steps.append({"t": dt, "running": running, "kv_blocks": in_use})
        if timing:
            ev = tcur.pop("_ev", [])
            if os.environ.get("VLLM_ASYNC_SCHED") == "1":
                # Async scheduling: the step returns before its forward finishes; read the events after the run.
                tdefer.append((len(trows), ev)); gpu_ms = 0.0
            else:
                gpu_ms = sum(a.elapsed_time(b) for a, b in ev) if ev else 0.0   # the step has synced; events are done
            trows.append([round(1000 * dt, 3), running or 0, tcur.get("reqs", 0), tcur.get("toks", 0),
                          tcur.get("code_rows", 0), tcur.get("code", 0), round(gpu_ms, 3)]
                         + [round(1000 * tcur.get(k, 0.0), 3) for k in TKEYS])
            tcur.clear()
        if profile:
            steps[-1]["sched"] = sched_log[:]; sched_log.clear()
        finished += sum(1 for o in outs if o.finished)
        if lengths is not None:
            now = time.perf_counter() - t_start
            for o in outs:
                if o.finished: finish_t[o.request_id] = round(now, 3)
        if lengths is not None and len(steps) % 2000 == 0:
            print(f"[bench] step {len(steps)} t={time.perf_counter() - t_start:.0f}s finished {finished}/{n_req} "
                  f"running {running} kv {in_use}/{total_blocks} preempt {preempt['events']} "
                  f"discarded {preempt['tokens_discarded']}", flush=True)
        if len(steps) == args.warmup_steps:
            t_steady = time.perf_counter()
            tokens_at_steady = sum(s["running"] or 0 for s in steps)
    wall = time.perf_counter() - t_start

    # decode tokens produced = steps x running sequences (forced length, one token per running seq per step)
    decode_tokens = sum(s["running"] or 0 for s in steps)
    # max_running is the EARLY peak: at step 1 every admitted request is a few hundred tokens long, so a saturation
    # run reports the request count, not how many FULL-LENGTH sequences the GPU sustains. Report the running count
    # at the step where the cache is fullest (sequences longest), and the median over the last quarter of the run.
    kv_steps = [s for s in steps if s["kv_blocks"] is not None]
    running_at_kv_peak = (max(kv_steps, key=lambda s: s["kv_blocks"])["running"] if kv_steps else None)
    tail = steps[int(0.75 * len(steps)):] or steps
    tail_running = sorted(s["running"] or 0 for s in tail)
    steady_running = tail_running[len(tail_running) // 2] if tail_running else None
    steady_steps = steps[args.warmup_steps:] if len(steps) > args.warmup_steps else steps
    steady_tokens = sum(s["running"] or 0 for s in steady_steps)
    steady_time = sum(s["t"] for s in steady_steps)
    per_step = sorted(s["t"] for s in steady_steps)
    rec = {
        "tag": args.tag, "mode": args.mode, "len": args.length, "prompt_len": args.prompt_len,
        "batch": args.batch, "requests": n_req, "rep": args.rep,
        "max_num_seqs": args.max_num_seqs, "max_model_len": args.max_model_len,
        "wall_s": round(wall, 3), "steps": len(steps), "finished": finished, "truncated": truncated,
        "decode_tokens": decode_tokens,
        "tok_per_s": round(decode_tokens / wall, 1) if wall else None,
        "steady_tok_per_s": round(steady_tokens / steady_time, 1) if steady_time else None,
        "step_ms_median": round(1000 * per_step[len(per_step) // 2], 3) if per_step else None,
        "max_running": max_running,
        "running_at_kv_peak": running_at_kv_peak, "steady_running": steady_running,
        "kv_blocks_peak": kv_peak, "kv_blocks_total": total_blocks, "kv_block_size": block_size,
        "kv_cache_groups": n_groups,
        "kv_group_blocks_peak": group_blocks_peak, "kv_group_quotas": group_quotas,
        "kv_bytes_peak": None, "kv_bytes_per_seq": None,
        "compresslm": bool(compress_config),
        "physical_quotas": bool(compress_config and compress_config.uses_physical_layer_quotas),
        "layer_quotas": bool(compress_config and compress_config.uses_layer_quotas),
        "env": {k: os.environ.get(k) for k in ("COMPRESSLM_PHYSICAL_QUOTAS", "COMPRESSLM_LAYER_PIECEWISE", "VLLM_GPU_MEM_UTIL")},
        # per-step latency vs. position (subsampled so the record stays small)
        "step_ms_trace": [round(1000 * s["t"], 3) for s in steps[::max(1, len(steps) // 256)]],
        "kv_blocks_trace": [s["kv_blocks"] for s in steps[::max(1, len(steps) // 256)]],
        "running_trace": [s["running"] for s in steps[::max(1, len(steps) // 256)]],
    }
    if profile:
        code = [x for x in steps if any(e[0] for e in x.get("sched", []))]
        dec = [x for x in steps if x.get("sched") and not any(e[0] for e in x["sched"])]
        n_sched = sum(len(x.get("sched", [])) for x in steps)
        code_reqs = [e[1] for x in code for e in x["sched"] if e[0]]
        code_toks = [e[2] for x in code for e in x["sched"] if e[0]]
        dec_reqs = [e[1] for x in dec for e in x["sched"]]
        t_code = sum(x["t"] for x in code); t_dec = sum(x["t"] for x in dec)
        # request-steps lost: every running request NOT in a code step waits for it
        idle = sum(max(0, (x["running"] or 0) - sum(e[1] for e in x["sched"] if e[0])) for x in code)
        mean = lambda v: (sum(v) / len(v)) if v else None
        rec["step_profile"] = {
            "steps": len(steps), "schedule_calls": n_sched, "code_steps": len(code), "decode_steps": len(dec),
            "wall_code_s": round(t_code, 2), "wall_decode_s": round(t_dec, 2),
            "frac_wall_code": round(t_code / wall, 4) if wall else None,
            "mean_code_step_ms": round(1000 * t_code / len(code), 3) if code else None,
            "mean_decode_step_ms": round(1000 * t_dec / len(dec), 3) if dec else None,
            "mean_reqs_per_code_step": mean(code_reqs), "mean_tokens_per_code_step": mean(code_toks),
            "mean_reqs_per_decode_step": mean(dec_reqs),
            "idle_request_steps_in_code_steps": idle,
            "decode_steps_per_code_step": round(len(dec) / len(code), 3) if code else None,
        }
        # Decode steps right after a code step pay for re-adding the requests the code step left out of the
        # persistent batch; split them out so that cost is not hidden in the plain decode-step mean.
        is_code = [any(e[0] for e in x.get("sched", [])) for x in steps]
        after = [steps[i]["t"] for i in range(1, len(steps))
                 if is_code[i - 1] and not is_code[i] and steps[i].get("sched")]
        other = [steps[i]["t"] for i in range(len(steps)) if not is_code[i] and steps[i].get("sched")
                 and not (i and is_code[i - 1])]
        rec["step_profile"].update({
            "mean_decode_step_after_code_ms": round(1000 * mean(after), 3) if after else None,
            "mean_decode_step_other_ms": round(1000 * mean(other), 3) if other else None,
        })
        try:   # in-process engine only (VLLM_ENABLE_V1_MULTIPROCESSING=0)
            runner = engine.engine_core.engine_core.model_executor.driver_worker.worker.model_runner
            rec["step_profile"]["code_step_graph_modes"] = dict(getattr(runner, "compresslm_code_step_graph_modes", {}))
        except AttributeError:
            pass
        ready = sorted(e[3] for x in code for e in x["sched"] if e[0])
        pct = lambda q: ready[min(len(ready) - 1, int(q * len(ready)))] if ready else None
        rec["step_profile"].update({
            "code_ready_per_code_step": {"mean": mean(ready), "p50": pct(0.5), "p90": pct(0.9), "p99": pct(0.99),
                                         "max": ready[-1] if ready else None},
            "mean_code_reqs_per_code_step": mean([e[4] for x in code for e in x["sched"] if e[0]]),
            "mean_decodes_per_code_step": mean([e[1] - e[4] for x in code for e in x["sched"] if e[0]]),
            "deferred_code_request_steps": sum(e[3] - e[4] for x in code for e in x["sched"] if e[0]),
        })
        print(f"[profile] {json.dumps(rec['step_profile'])}", flush=True)
    if lengths is not None:
        ft = sorted(finish_t.values())
        rec.update({
            "lengths_file": args.lengths_file, "lengths_key": args.lengths_key, "len_scale": args.len_scale,
            "real_prompts": bool(args.real_prompts),
            "sampling": ({"temperature": args.sample_temperature, "top_p": args.sample_top_p, "top_k": args.sample_top_k,
                          "seed": args.sample_seed} if args.real_prompts else "greedy"),
            "mean_len": sum(lengths) / len(lengths), "max_len": max(lengths), "gen_tokens_total": sum(lengths),
            # headline: GPU-seconds per problem = time to finish the whole workload / problems in it
            "s_per_problem": round(wall / n_req, 4),
            # drain-insensitive variant: time until 90% of requests are done, per finished request
            "t90_s": ft[int(0.9 * len(ft)) - 1] if len(ft) >= 10 else None,
            "finish_t": [finish_t.get(str(i)) for i in range(n_req)],
            "all_finished": finished == n_req,
            "preemptions": preempt["events"], "preempt_tokens_discarded": preempt["tokens_discarded"],
        })
    if os.environ.get("COMPRESSLM_COUNT_BLOCKS") == "1":
        rec["blocks_finished"] = blocks["finished"]
        rec["visible_tokens_per_block"] = round(generated / blocks["finished"], 2) if blocks["finished"] else None
        print(f"[bench] blocks finished {blocks['finished']} over {generated} visible tokens "
              f"({rec['visible_tokens_per_block']} per block)", flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    name = f"{args.tag}_{args.mode}_B{args.batch if args.mode == 'batch' else n_req}_L{args.length}_r{args.rep}.json"
    if lengths is not None:
        name = f"{args.tag}_complete_N{n_req}_S{args.len_scale:g}_M{args.max_num_seqs}_r{args.rep}.json"
    # code-step counters of the COMPRESSLM_CODE_BATCH scheduler policy (third_party/vllm_overlay_skip); absent otherwise
    _s = getattr(getattr(engine.engine_core, "engine_core", None), "scheduler", None)
    if getattr(_s, "compresslm_code_steps", None) is not None:
        rec.update({"code_batch": os.environ.get("COMPRESSLM_CODE_BATCH"),
                    "code_batch_maxwait": os.environ.get("COMPRESSLM_CODE_BATCH_MAXWAIT", "64"),
                    "sched_steps": _s.compresslm_sched_steps, "code_steps": _s.compresslm_code_steps,
                    "code_reqs_per_code_step": round(_s.compresslm_code_rows / max(1, _s.compresslm_code_steps), 2)})
        print(f"[bench] code steps {rec['code_steps']} of {rec['sched_steps']} scheduler steps, "
              f"{rec['code_reqs_per_code_step']} requests per code step (g={rec['code_batch']})", flush=True)
    if timing and tdefer:
        import torch
        torch.cuda.synchronize()
        for i, ev in tdefer:
            trows[i][6] = round(sum(a.elapsed_time(b) for a, b in ev), 3) if ev else 0.0
    if timing and trows:
        cols = ["wall_ms", "running", "reqs", "toks", "code_rows", "code_step", "fwd_gpu_ms"] + [k + "_ms" for k in TKEYS]
        Path(args.out, name.replace(".json", ".steptiming.json")).write_text(json.dumps({"cols": cols, "rows": trows}))
        def _agg(rows):
            if not rows: return None
            m = {c: round(sum(r[i] for r in rows) / len(rows), 3) for i, c in enumerate(cols)}
            m["n"] = len(rows); m["total_s"] = round(sum(r[0] for r in rows) / 1000, 2)
            # host = wall - forward GPU span, i.e. everything the GPU is not running the model for
            m["host_ms"] = round(m["wall_ms"] - m["fwd_gpu_ms"], 3)
            return m
        dec = [r for r in trows if not r[5] and r[4] == 0 and r[2] > 0]
        code = [r for r in trows if r[5] or r[4] > 0]
        bins = {}
        for r in dec:
            bins.setdefault(min(r[1] // 16, 12) * 16, []).append(r)
        rec["step_timing"] = {"all": _agg(trows), "decode": _agg(dec), "code": _agg(code),
                              "decode_by_running": {str(b): _agg(v) for b, v in sorted(bins.items())}}
        print(f"[timing] {json.dumps({k: rec['step_timing'][k] for k in ('all', 'decode', 'code')})}", flush=True)
    Path(args.out, name).write_text(json.dumps(rec, indent=2))
    if group_blocks_peak:   # printed first: the launcher reports the LAST [bench] line as the cell's result
        print(f"[bench] per-group blocks {group_blocks_peak} for quotas {group_quotas}", flush=True)
    if lengths is not None:
        print(f"[bench] {name}: s/problem {rec['s_per_problem']} wall {rec['wall_s']} finished {finished}/{n_req} "
              f"mean_len {rec['mean_len']:.0f} tok/s {rec['tok_per_s']} preempt {preempt['events']} "
              f"discarded {preempt['tokens_discarded']}", flush=True)
    print(f"[bench] {name}: tok/s {rec['tok_per_s']} steady {rec['steady_tok_per_s']} step_ms {rec['step_ms_median']} "
          f"max_running {max_running} (at kv peak {running_at_kv_peak}, steady {steady_running}) "
          f"kv_blocks {kv_peak}/{total_blocks} (block {block_size}, groups {n_groups})", flush=True)


if __name__ == "__main__":
    main()
    os._exit(0)     # vLLM shutdown can hang after results are written (see scripts/eval_aime.py)
