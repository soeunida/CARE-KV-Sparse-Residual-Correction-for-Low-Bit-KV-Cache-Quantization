"""tools/bench_prefill_decode_perf.py

Prefill / decode performance benchmark for CARE-KV vs baselines.

Measures, separately and with CUDA events:
  - prefill latency        (one forward on [batch, seq_len], use_cache=True)
  - decode latency         (token-by-token over gen_tokens, reusing the cache)
  - TTFT                    (prefill + first decode step)
  - throughput             (prefill tok/s; decode tok/s EXCLUDING prefill)
  - runtime peak memory     (max_memory_allocated / max_memory_reserved over the
                             full run — weights + activations + cache, NOT a
                             clean KV-only allocation)

Methods (reusing baselines/ adapters):
  fp16, basequant_int3, basequant_int4, adaptive_carekv_int3,
  turboquant_int3, turboquant_int4.
INT4 is a higher-bit reference only. TurboQuant is standalone-only;
TurboQuant+CARE-KV is unsupported (one preserved marker row per model).

Determinism: TF32 off, PYTHONHASHSEED=0, fixed seeds. Warmup >=2, measured >=5
reps; mean/std/min/max reported. Failed/OOM/unsupported rows are preserved.

Decode driving: a real HF DynamicCache + cache_position is threaded through
prefill and every decode step — exactly what generate() does internally — so the
CARE-KV patched attention (which reads past_key_value.get_seq_length() to detect
decode) tracks page growth correctly, and TurboQuant/fp16 use the standard cache.
"""
from __future__ import annotations

import os
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import csv
import gc
import math
import statistics
import traceback

import torch
from transformers import AutoTokenizer, DynamicCache

from CARE_KV.care_kv import reset_all_caches
from CARE_KV.care_kv.baselines.fp16_adapter import FP16Adapter
from CARE_KV.care_kv.baselines.basequant_adapter import BaseQuantAdapter
from CARE_KV.care_kv.baselines.carekv_adapter import CAREKVAdapter
from CARE_KV.care_kv.baselines.turboquant_style import TurboQuantStyleAdapter

MB = 1024 ** 2

MODELS = [
    "mistralai/Mistral-7B-v0.3",
    "01-ai/Yi-6B",
    "deepseek-ai/deepseek-llm-7b-base",
    "openlm-research/open_llama_7b_v2",
]

from transformers import LlamaForCausalLM


class _FP16BackendAdapter(FP16Adapter):
    """FP16 reference with an explicit attention backend (sdpa | eager).

    The plain `fp16` method keeps HF's default (sdpa on Llama/Mistral); these
    let us run an explicit SDPA-optimized baseline AND a same-backend eager
    control so TurboQuant (eager-only) is compared fairly."""
    def __init__(self, attn_impl: str):
        self._attn_impl = attn_impl
        self.name = f"fp16_{attn_impl}"
        self.bit_width = "fp16"

    def setup_model(self, model_id: str):
        torch.manual_seed(0)
        m = LlamaForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16,
            attn_implementation=self._attn_impl, device_map="cuda")
        m.config.use_cache = False
        m.eval()
        return m


# method -> (adapter factory, bit_width, is_patched_carekv_cache, higher_bit_ref)
METHODS = {
    "fp16":                 (lambda: FP16Adapter(),                       "16", False, False),
    "fp16_sdpa":            (lambda: _FP16BackendAdapter("sdpa"),         "16", False, False),
    "fp16_eager":           (lambda: _FP16BackendAdapter("eager"),        "16", False, False),
    "basequant_int3":       (lambda: BaseQuantAdapter(bits=3),            "3",  True,  False),
    "basequant_int4":       (lambda: BaseQuantAdapter(bits=4),            "4",  True,  True),
    "adaptive_carekv_int3": (lambda: CAREKVAdapter(),                     "3",  True,  False),
    "turboquant_int3":      (lambda: TurboQuantStyleAdapter(bits_k=3, bits_v=3), "3", False, False),
    "turboquant_int4":      (lambda: TurboQuantStyleAdapter(bits_k=4, bits_v=4), "4", False, True),
}

# backend-fairness metadata: method -> (backend_group, fair_runtime_group)
BACKEND_META = {
    "fp16":                 ("sdpa",        "sdpa_optimized_baseline"),
    "fp16_sdpa":            ("sdpa",        "sdpa_optimized_baseline"),
    "fp16_eager":           ("eager",       "eager_backend_comparison"),
    "turboquant_int3":      ("eager",       "eager_backend_comparison"),
    "turboquant_int4":      ("eager",       "eager_backend_comparison"),
    "basequant_int3":       ("python_loop", "prototype_python_loop_blocker"),
    "basequant_int4":       ("python_loop", "prototype_python_loop_blocker"),
    "adaptive_carekv_int3": ("python_loop", "prototype_python_loop_blocker"),
    "turboquant_plus_carekv": ("n/a",       "unsupported"),
}


def backend_meta(method):
    return BACKEND_META.get(method, ("", ""))

CSV_COLS = [
    "model_id", "method", "bit_width", "seq_len", "batch_size", "gen_tokens",
    "selected_correction", "effective_budget",
    "prefill_ms_mean", "prefill_ms_std", "prefill_tok_per_s_mean",
    "decode_total_ms_mean", "decode_ms_per_token_mean", "decode_tok_per_s_mean",
    "ttft_ms_mean", "peak_allocated_MB", "peak_reserved_MB", "kv_cache_MB",
    "ppl", "status", "failure_reason", "oom", "notes",
    # runtime-scope labelling (slow_micro diagnostic vs fast_full measured)
    "runtime_scope", "runtime_status",
    # backend-fairness labelling (sdpa vs eager vs python_loop)
    "backend_group", "fair_runtime_group",
    # extra detail (kept beyond the required columns)
    "prefill_ms_min", "prefill_ms_max", "decode_total_ms_std",
    "ttft_ms_std", "n_warmup", "n_reps", "attn_impl", "scope",
]

# Tiered scope (decided with the user): the CARE-KV / BaseQuant patched decode
# is a per-(layer,kv_head,t) Python-loop prototype (~11 s/token), ~290x slower
# than fp16, so the full grid is infeasible for them. fp16 + TurboQuant are fast.
#   fast tier  -> full grid, 5 reps
#   slow tier  -> reduced grid (batch=1, seq<=512, gen=32), 3 reps
FAST_METHODS = {"fp16", "turboquant_int3", "turboquant_int4"}
SLOW_METHODS = {"basequant_int3", "basequant_int4", "adaptive_carekv_int3"}


def get_selected_correction_map(path):
    """Map (model_id, seq_len) -> (selected_correction, effective_budget) from the
    prior selector data, if available."""
    m = {}
    if not os.path.exists(path):
        return m
    try:
        import pandas as pd
        df = pd.read_csv(path)
        ck = df[df["method"] == "Adaptive_CAREKV_INT3"]
        for _, r in ck.iterrows():
            sc = str(r.get("selected_correction"))
            eff = ("SK0SV4" if sc == "Vdom" else "SK2SV4" if sc == "KV" else "SK0SV0")
            m[(r["model_id"], int(r["seq_len"]))] = (sc, eff)
    except Exception as e:
        print(f"[selector-map] skipped: {e}")
    return m


def effective_budget_for(method, model_id, seq_len, sel_map):
    if method == "adaptive_carekv_int3":
        sc, eff = sel_map.get((model_id, seq_len), ("(uncalibrated)", "SK2SV4"))
        return sc, eff
    if method.startswith("basequant"):
        return "", "SK0SV0"          # base quant, no residuals
    if method.startswith("turboquant"):
        return "", "standalone"
    return "", ""                    # fp16


def time_prefill_decode(model, input_ids, gen_tokens, is_carekv_cache, device):
    """One full prefill+decode pass. Returns (t_prefill_ms, t_first_ms, t_decode_total_ms)."""
    B, S = input_ids.shape
    if is_carekv_cache:
        reset_all_caches(model)
    cache = DynamicCache()
    pos = torch.arange(S, device=device)
    pid = pos.unsqueeze(0).expand(B, -1)
    am = torch.ones(B, S, device=device, dtype=torch.long)

    pe0 = torch.cuda.Event(enable_timing=True)
    pe1 = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    pe0.record()
    out = model(input_ids=input_ids, attention_mask=am, position_ids=pid,
                past_key_values=cache, use_cache=True, cache_position=pos)
    pe1.record()
    nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
    cur = S

    dec_events = []
    for _ in range(gen_tokens):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        cp = torch.tensor([cur], device=device)
        pid_d = torch.full((B, 1), cur, device=device, dtype=torch.long)
        am = torch.ones(B, cur + 1, device=device, dtype=torch.long)
        s.record()
        out = model(input_ids=nxt, attention_mask=am, position_ids=pid_d,
                    past_key_values=cache, use_cache=True, cache_position=cp)
        nxt = out.logits[:, -1, :].argmax(-1, keepdim=True)
        e.record()
        dec_events.append((s, e))
        cur += 1
    torch.cuda.synchronize()
    t_prefill = pe0.elapsed_time(pe1)
    step_ms = [s.elapsed_time(e) for s, e in dec_events]
    return t_prefill, step_ms[0], float(sum(step_ms))


def bench_config(model, tok, arch, method, mid, seq_len, batch, gen_tokens,
                 is_carekv_cache, n_warmup, n_reps, device, sel_map,
                 bit_width, higher_ref, attn_impl, scope="",
                 runtime_scope="", runtime_status="", notes_override=None):
    sc, eff = effective_budget_for(method, mid, seq_len, sel_map)
    kvmb = ""
    base_notes = notes_override if notes_override is not None else (
        "higher-bit reference" if higher_ref else "")
    bgrp, frgrp = backend_meta(method)
    row = dict(model_id=mid, method=method, bit_width=bit_width, seq_len=seq_len,
               batch_size=batch, gen_tokens=gen_tokens, selected_correction=sc,
               effective_budget=eff, ppl="", n_warmup=n_warmup, n_reps=n_reps,
               attn_impl=attn_impl, oom="no", failure_reason="", scope=scope,
               runtime_scope=runtime_scope, runtime_status=runtime_status,
               backend_group=bgrp, fair_runtime_group=frgrp,
               notes=base_notes)
    # deterministic input tokens (fixed seed; avoid pad)
    g = torch.Generator(device="cpu").manual_seed(1234)
    vocab = int(getattr(arch, "vocab_size", 32000))
    ids = torch.randint(10, max(11, vocab - 1), (batch, seq_len), generator=g).to(device)
    try:
        # analytical KV estimate (method's theoretical bytes — NOT the runtime peak)
        try:
            est = model_adapter_est.get(method)
            if est is not None:
                kvmb = round(est(seq_len, arch.num_hidden_layers,
                                 getattr(arch, "num_key_value_heads", arch.num_attention_heads),
                                 arch.hidden_size // arch.num_attention_heads)
                             .get("estimated_kv_memory_MB", 0.0) * batch, 4)
        except Exception:
            kvmb = ""
        row["kv_cache_MB"] = kvmb

        # warmup primes kernels/lazy-alloc only — use a SHORT gen so the slow
        # Python-loop methods don't pay full gen_tokens during warmup.
        gen_warm = min(gen_tokens, 4)
        with torch.no_grad():
            for _ in range(n_warmup):
                time_prefill_decode(model, ids, gen_warm, is_carekv_cache, device)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            pref, ttft, dtot = [], [], []
            for _ in range(n_reps):
                tp, tf, td = time_prefill_decode(model, ids, gen_tokens,
                                                 is_carekv_cache, device)
                pref.append(tp); ttft.append(tp + tf); dtot.append(td)
            peak_alloc = torch.cuda.max_memory_allocated() / MB
            peak_resv = torch.cuda.max_memory_reserved() / MB

        def ms(x): return round(statistics.mean(x), 4)
        def sd(x): return round(statistics.pstdev(x), 4) if len(x) > 1 else 0.0
        dpt = [d / gen_tokens for d in dtot]
        row.update(
            prefill_ms_mean=ms(pref), prefill_ms_std=sd(pref),
            prefill_ms_min=round(min(pref), 4), prefill_ms_max=round(max(pref), 4),
            prefill_tok_per_s_mean=round(batch * seq_len * 1000.0 / statistics.mean(pref), 2),
            decode_total_ms_mean=ms(dtot), decode_total_ms_std=sd(dtot),
            decode_ms_per_token_mean=ms(dpt),
            decode_tok_per_s_mean=round(batch * gen_tokens * 1000.0 / statistics.mean(dtot), 2),
            ttft_ms_mean=ms(ttft), ttft_ms_std=sd(ttft),
            peak_allocated_MB=round(peak_alloc, 2), peak_reserved_MB=round(peak_resv, 2),
            status="ok",
        )
    except torch.cuda.OutOfMemoryError as e:
        gc.collect(); torch.cuda.empty_cache()
        row.update(status="oom", oom="yes", failure_reason=repr(e)[:160],
                   kv_cache_MB=kvmb)
    except Exception as e:
        gc.collect(); torch.cuda.empty_cache()
        row.update(status="failed", failure_reason=(repr(e)[:200]), kv_cache_MB=kvmb)
        traceback.print_exc()
    for c in CSV_COLS:
        row.setdefault(c, "")
    return row


# adapter memory estimators, filled at runtime per method
model_adapter_est = {}


def _resume_seed(out_dir, smoke_tag):
    """Read an existing all_rows CSV and return (seed_rows, done_ok, marker_models).

    done_ok: set of (model_id, method, seq_len, batch_size, gen_tokens) already ok.
    marker_models: models that already have the unsupported turboquant+carekv row.
    Lets repeated relaunches (this box SIGKILLs long jobs) accumulate instead of
    re-running completed configs."""
    import pandas as pd
    path = os.path.join(out_dir, f"prefill_decode_all_rows{smoke_tag}.csv")
    if not os.path.exists(path):
        return [], set(), set()
    df = pd.read_csv(path)
    seed, done, markers = [], set(), set()
    def _k(v):
        try: return int(float(v))
        except (TypeError, ValueError): return v
    for _, r in df.iterrows():
        d = {c: r[c] for c in df.columns if c in CSV_COLS}
        st = str(r.get("status"))
        if r.get("method") == "turboquant_plus_carekv":
            seed.append(d); markers.add(r["model_id"]); continue
        if st == "ok":
            seed.append(d)
            done.add((r["model_id"], r["method"], _k(r["seq_len"]),
                      _k(r["batch_size"]), _k(r["gen_tokens"])))
    return seed, done, markers


def _add_marker(all_rows, mid):
    unsup = dict(model_id=mid, method="turboquant_plus_carekv", bit_width="3",
                 seq_len="", batch_size="", gen_tokens="", selected_correction="",
                 effective_budget="unsupported", ppl="", status="unsupported",
                 oom="no", scope="n/a",
                 failure_reason="QJL is a score-level inner-product estimator; "
                 "CARE-KV corrects reconstructed K/V values — stacking redefines "
                 "the methods", notes="standalone-only; never combined",
                 backend_group="n/a", fair_runtime_group="unsupported")
    for c in CSV_COLS:
        unsup.setdefault(c, "")
    all_rows.append(unsup)


def run(models, method_grids, out_dir, smoke_tag="", guard=None, resume=True):
    """method_grids: dict method -> dict(seqlens, batches, gens, n_warmup, n_reps, scope).
    guard: optional dict(max_ms, policy in {run,skip,micro_only}) — the slow-method
    runtime guard. Once a method's measured decode_ms_per_token exceeds max_ms (or
    policy=micro_only after the first config), later rows for that method are
    recorded `skipped_prototype_runtime_infeasible` /
    `python_loop_decode_path_too_slow` instead of being run."""
    guard = guard or {}
    g_max = guard.get("max_ms"); g_policy = guard.get("policy", "run")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    sel_map = get_selected_correction_map(
        "results/memory_aware_fair_comparison/memory_aware_all_rows.csv")

    # bound the CARE-KV/BaseQuant fp16 side-buffer for the largest (seq+gen)
    # across ALL method grids.
    max_tokens = max(max(g["seqlens"]) + max(g["gens"]) for g in method_grids.values())
    os.environ["CAREKV_MAX_PAGES"] = str(math.ceil(max_tokens / 16) + 8)

    all_rows, done_ok, marker_models = ([], set(), set())
    if resume:
        all_rows, done_ok, marker_models = _resume_seed(out_dir, smoke_tag)
        if all_rows:
            print(f"[resume] seeded {len(all_rows)} prior rows, "
                  f"{len(done_ok)} completed configs will be skipped", flush=True)

    for mid in models:
        # unsupported marker (preserved), once per model (skip if already seeded)
        if mid not in marker_models:
            _add_marker(all_rows, mid)
            marker_models.add(mid)

        for method, (factory, bit_width, is_cc, higher_ref) in METHODS.items():
            if method not in method_grids:
                continue
            grid = method_grids[method]
            # resume: if every config for this (model,method) is already done,
            # skip loading the 7B model entirely.
            all_done = all((mid, method, int(S), int(B), int(G)) in done_ok
                           for S in grid["seqlens"] for B in grid["batches"]
                           for G in grid["gens"])
            if all_done:
                print(f"=== {mid} | {method}: all configs done (resume-skip load) ===",
                      flush=True)
                continue
            print(f"\n=== {mid} | {method} (load) [{grid['scope']}] ===", flush=True)
            adapter = factory()
            model = None
            try:
                model = adapter.setup_model(mid)
                model.config.use_cache = True
                model.eval()
                arch = model.config
                attn_impl = getattr(arch, "_attn_implementation", "?")
                model_adapter_est[method] = adapter.estimate_memory
            except Exception as e:
                is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
                gc.collect(); torch.cuda.empty_cache()
                print(f"[load-{'oom' if is_oom else 'failed'}] {mid} {method}: {e}")
                for S in grid["seqlens"]:
                    for B in grid["batches"]:
                        for G in grid["gens"]:
                            _bg, _fg = backend_meta(method)
                            r = dict(model_id=mid, method=method, bit_width=bit_width,
                                     seq_len=S, batch_size=B, gen_tokens=G,
                                     status=("oom" if is_oom else "failed"),
                                     oom=("yes" if is_oom else "no"), scope=grid["scope"],
                                     runtime_scope=grid.get("runtime_scope", ""),
                                     runtime_status=grid.get("runtime_status", ""),
                                     backend_group=_bg, fair_runtime_group=_fg,
                                     notes=(grid.get("notes_override") or ""),
                                     failure_reason=f"load: {repr(e)[:120]}")
                            for c in CSV_COLS: r.setdefault(c, "")
                            all_rows.append(r)
                write_outputs(all_rows, out_dir, smoke_tag)
                continue

            method_tripped = False   # slow-method guard state (per method)
            ran_one = False
            for S in grid["seqlens"]:
                for B in grid["batches"]:
                    for G in grid["gens"]:
                        # resume: already-completed config is in the seeded rows
                        if (mid, method, int(S), int(B), int(G)) in done_ok:
                            ran_one = True
                            print(f"  {method} SL{S} B{B} G{G} ... resume-skip (done)",
                                  flush=True)
                            continue
                        # slow-method guard: skip remaining configs once tripped
                        skip = False
                        if g_policy == "micro_only" and ran_one:
                            skip = True
                        elif method_tripped and g_policy in ("skip", "micro_only"):
                            skip = True
                        if skip:
                            _bg, _fg = backend_meta(method)
                            r = dict(model_id=mid, method=method, bit_width=bit_width,
                                     seq_len=S, batch_size=B, gen_tokens=G,
                                     status="skipped_prototype_runtime_infeasible", oom="no",
                                     scope=grid["scope"], runtime_scope=grid.get("runtime_scope", ""),
                                     runtime_status=grid.get("runtime_status", ""),
                                     backend_group=_bg, fair_runtime_group=_fg,
                                     failure_reason="python_loop_decode_path_too_slow",
                                     notes=(grid.get("notes_override") or ""))
                            for c in CSV_COLS: r.setdefault(c, "")
                            all_rows.append(r)
                            print(f"  {method} SL{S} B{B} G{G} ... SKIPPED (guard: {g_policy})", flush=True)
                            write_outputs(all_rows, out_dir, smoke_tag); continue
                        print(f"  {method} SL{S} B{B} G{G} ...", end="", flush=True)
                        r = bench_config(model, tok=None, arch=arch, method=method,
                                         mid=mid, seq_len=S, batch=B, gen_tokens=G,
                                         is_carekv_cache=is_cc,
                                         n_warmup=grid["n_warmup"], n_reps=grid["n_reps"],
                                         device=device, sel_map=sel_map,
                                         bit_width=bit_width, higher_ref=higher_ref,
                                         attn_impl=attn_impl, scope=grid["scope"],
                                         runtime_scope=grid.get("runtime_scope", ""),
                                         runtime_status=grid.get("runtime_status", ""),
                                         notes_override=grid.get("notes_override"))
                        all_rows.append(r); ran_one = True
                        print(f" {r['status']}"
                              + (f" pf={r['prefill_ms_mean']}ms dec/t={r['decode_ms_per_token_mean']}ms"
                                 f" peak={r['peak_allocated_MB']}MB" if r["status"] == "ok" else ""),
                              flush=True)
                        # trip the guard if this config exceeded the threshold
                        if g_max is not None and r["status"] == "ok":
                            dmt = r.get("decode_ms_per_token_mean")
                            try:
                                if dmt is not None and float(dmt) > float(g_max):
                                    method_tripped = True
                                    print(f"  [guard] {method} decode {dmt} ms/tok > {g_max} ms "
                                          f"→ later {method} rows skipped (policy={g_policy})", flush=True)
                            except (TypeError, ValueError):
                                pass
                        if is_cc:
                            reset_all_caches(model)
                        gc.collect(); torch.cuda.empty_cache()
                        write_outputs(all_rows, out_dir, smoke_tag)   # incremental

            try:
                if hasattr(adapter, "teardown"):
                    adapter.teardown()
            except Exception:
                pass
            del model, adapter
            gc.collect(); torch.cuda.empty_cache()

    write_outputs(all_rows, out_dir, smoke_tag)
    return all_rows


def write_outputs(rows, out_dir, smoke_tag):
    os.makedirs(out_dir, exist_ok=True)
    P = lambda n: os.path.join(out_dir, n)
    suffix = smoke_tag

    def dump(path, rs):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
            w.writeheader()
            for r in rs:
                w.writerow(r)

    dump(P(f"prefill_decode_all_rows{suffix}.csv"), rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    dump(P(f"prefill_decode_success_only{suffix}.csv"), ok)
    dump(P(f"prefill_decode_failure_log{suffix}.csv"), fail)

    # summary by method (means over ok rows)
    import pandas as pd
    summ_path = P(f"prefill_decode_summary_by_method{suffix}.csv")
    if ok:
        df = pd.DataFrame(ok)
        for c in ["prefill_ms_mean", "decode_ms_per_token_mean", "decode_tok_per_s_mean",
                  "ttft_ms_mean", "peak_allocated_MB", "peak_reserved_MB"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        summ = (df.groupby(["method", "bit_width"], as_index=False)
                .agg(n_ok=("status", "size"),
                     prefill_ms_mean=("prefill_ms_mean", "mean"),
                     decode_ms_per_token_mean=("decode_ms_per_token_mean", "mean"),
                     decode_tok_per_s_mean=("decode_tok_per_s_mean", "mean"),
                     ttft_ms_mean=("ttft_ms_mean", "mean"),
                     peak_allocated_MB_mean=("peak_allocated_MB", "mean"),
                     peak_reserved_MB_mean=("peak_reserved_MB", "mean"))
                .sort_values("decode_ms_per_token_mean"))
        summ.to_csv(summ_path, index=False)
    else:
        pd.DataFrame(columns=["method"]).to_csv(summ_path, index=False)

    write_report(P(f"PREFILL_DECODE_PERF_REPORT{suffix}.md"), rows, ok, fail, summ_path)
    return summ_path


def write_report(path, rows, ok, fail, summ_path):
    import pandas as pd
    L = []
    L.append("# CARE-KV Prefill / Decode Performance Benchmark\n")
    L.append("Per-phase latency, throughput and **runtime peak memory** for "
             "fp16, BaseQuant INT3/INT4, Adaptive CARE-KV INT3, and TurboQuant "
             "INT3/INT4. INT4 rows are **higher-bit references**; TurboQuant is "
             "standalone-only; TurboQuant+CARE-KV is **unsupported** (preserved "
             "marker rows).\n")
    L.append("## Method / measurement notes\n")
    L.append("- Timing: CUDA events + `torch.cuda.synchronize()`. Determinism: "
             "TF32 off, `PYTHONHASHSEED=0`, fixed seeds.")
    L.append("- **Prefill**: one forward on `[batch, seq_len]` with "
             "`use_cache=True` (DynamicCache + cache_position).")
    L.append("- **Decode**: token-by-token over `gen_tokens`, reusing the "
             "produced cache. **decode throughput excludes prefill.**")
    L.append("- **TTFT** = prefill + first decode step.")
    L.append("- **peak_allocated_MB / peak_reserved_MB**: "
             "`torch.cuda.max_memory_allocated/reserved` over the full run "
             "(weights + activations + cache) — **not** a clean KV-only "
             "allocation.")
    L.append("- **kv_cache_MB**: the method's *analytical* theoretical KV bytes "
             "(adapter estimate × batch); for CARE-KV/BaseQuant the prototype "
             "fp16 side-buffer is excluded here but **is** in peak_allocated_MB.")
    L.append("- `ppl` left blank — this is a perf benchmark (PPL lives in the "
             "fair-comparison report).")
    L.append("- For CARE-KV, `selected_correction`/`effective_budget` are looked "
             "up from the selector data (Vdom→SK0SV4, KV→SK2SV4, "
             "skip/BaseQuant→SK0SV0); the measured run uses the fixed SK2SV4 "
             "carekv_stored runtime.\n")

    if os.path.exists(summ_path):
        try:
            s = pd.read_csv(summ_path)
            if len(s):
                L.append("## Summary by method (mean over successful configs)\n")
                d = s.copy()
                for c in d.columns:
                    if d[c].dtype.kind in "fc":
                        d[c] = d[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
                L.append("| " + " | ".join(d.columns) + " |")
                L.append("| " + " | ".join("---" for _ in d.columns) + " |")
                for r in d.itertuples(index=False):
                    L.append("| " + " | ".join(map(str, r)) + " |")
                L.append("")
        except Exception:
            pass

    L.append("## Coverage\n")
    L.append(f"- total rows: {len(rows)}  (ok: {len(ok)}, "
             f"non-ok preserved: {len(fail)})")
    from collections import Counter
    cs = Counter(r.get("status") for r in rows)
    L.append("- status counts: " + ", ".join(f"{k}={v}" for k, v in sorted(cs.items())))
    L.append("")
    L.append("## Failure / OOM / unsupported log\n")
    if fail:
        byreason = Counter((r.get("method"), r.get("status")) for r in fail)
        L.append("| method | status | count |")
        L.append("| --- | --- | --- |")
        for (m, st), c in sorted(byreason.items()):
            L.append(f"| {m} | {st} | {c} |")
    else:
        L.append("_No failures._")
    L.append("")
    L.append("## Honesty / preservation\n")
    L.append("- INT4 is a higher-bit reference, not ranked against INT3 as a winner.")
    L.append("- TurboQuant rows are standalone-only; TurboQuant+CARE-KV stays "
             "unsupported (marker rows preserved).")
    L.append("- OOM / failed / unsupported configurations are preserved in "
             "`prefill_decode_all_rows*.csv` and `prefill_decode_failure_log*.csv`.")
    L.append("- Peak memory is the full runtime peak; kv_cache_MB is analytical "
             "(separate, labelled).")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")


def build_tiered_grids():
    """fast tier (fp16 + TurboQuant): full grid, 5 reps / 2 warmup.
    slow tier (BaseQuant + CARE-KV prototype Python-loop): reduced grid,
    batch=1, seq<=512, gen=32, 3 reps / 1 warmup (short warmup gen)."""
    grids = {}
    for m in FAST_METHODS:
        grids[m] = dict(seqlens=[128, 256, 512, 1024], batches=[1, 2, 4],
                        gens=[32, 128], n_warmup=2, n_reps=5,
                        scope="fast_full(seq128-1024,batch1-4,gen32/128,reps5)")
    for m in SLOW_METHODS:
        grids[m] = dict(seqlens=[128, 256, 512], batches=[1], gens=[32],
                        n_warmup=1, n_reps=3,
                        scope="slow_reduced(seq128-512,batch1,gen32,reps3) prototype-Python-loop")
    return grids


SLOW_MICRO_NOTE = ("Current patched cache path is dominated by per-layer/"
                   "per-head/per-token Python loops and should not be "
                   "interpreted as optimized algorithmic runtime.")


TURBOQUANT_FAIRNESS_NOTE = (
    "TurboQuant reduces KV-cache component memory, but in the current "
    "implementation it runs through an eager attention path. We therefore "
    "include an FP16-eager control and report optimized FP16-SDPA separately.")


def write_backend_control_report(out_dir):
    """Backend-fairness report: A=fp16_sdpa optimized baseline, B=same-backend
    eager comparison (fp16_eager vs TurboQuant), C=prototype python-loop blocker
    (from slow_micro). Reads backend_control rows + slow_micro rows; never deletes."""
    import pandas as pd
    P = os.path.join(out_dir, "prefill_decode_all_rows.csv")
    if not os.path.exists(P):
        return
    df = pd.read_csv(P)
    ok = df[df["status"] == "ok"].copy()
    for c in ["prefill_ms_mean", "decode_ms_per_token_mean", "ttft_ms_mean",
              "peak_allocated_MB", "peak_reserved_MB"]:
        ok[c] = pd.to_numeric(ok[c], errors="coerce")

    L = ["# Backend-Control Runtime Benchmark (fairness)\n"]
    L.append(TURBOQUANT_FAIRNESS_NOTE + "\n")
    L.append("Methods are grouped by attention backend so runtime is compared "
             "**within the same backend**:")
    L.append("- `sdpa` → `sdpa_optimized_baseline` (fused FlashAttention kernel)")
    L.append("- `eager` → `eager_backend_comparison` (materialized-score path; "
             "fp16_eager and TurboQuant both live here)")
    L.append("- `python_loop` → `prototype_python_loop_blocker` (BaseQuant / "
             "CARE-KV patched cache; from slow_micro, not run here)\n")

    def agg(d):
        return d.groupby("method", as_index=False).agg(
            n=("status", "size"),
            prefill_ms=("prefill_ms_mean", "mean"),
            decode_ms_per_tok=("decode_ms_per_token_mean", "mean"),
            ttft_ms=("ttft_ms_mean", "mean"),
            peak_alloc_MB=("peak_allocated_MB", "mean"))

    def tbl(d, cols=None, fmt=2):
        d = d.copy()
        if cols: d = d[cols]
        for c in d.columns:
            if d[c].dtype.kind in "fc":
                d[c] = d[c].map(lambda v: f"{v:.{fmt}f}" if pd.notna(v) else "")
        return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |",
                          "| " + " | ".join("---" for _ in d.columns) + " |"]
                         + ["| " + " | ".join(map(str, r)) + " |"
                            for r in d.itertuples(index=False)])

    # A. optimized baseline
    L.append("## A. Optimized baseline — fp16_sdpa\n")
    A = ok[ok["method"] == "fp16_sdpa"]
    L.append(tbl(agg(A)) if len(A) else "_no fp16_sdpa rows_")
    L.append("\nThis is the production-optimized FP16 path (fused SDPA). It is "
             "**not** the fair comparator for an eager-only method — it is the "
             "ceiling, reported separately.\n")

    # B. same-backend eager comparison
    L.append("## B. Same-backend eager comparison — fp16_eager vs TurboQuant\n")
    B = ok[ok["backend_group"] == "eager"]
    L.append(tbl(agg(B)))
    # fair per-config slowdown vs fp16_eager
    base = (B[B["method"] == "fp16_eager"]
            .set_index(["model_id", "seq_len", "batch_size", "gen_tokens"])
            ["decode_ms_per_token_mean"])
    rels = []
    for m in ("turboquant_int3", "turboquant_int4"):
        sub = B[B["method"] == m].set_index(
            ["model_id", "seq_len", "batch_size", "gen_tokens"])["decode_ms_per_token_mean"]
        j = (sub / base).dropna()
        if len(j):
            rels.append((m, round(j.mean(), 3), round(j.min(), 3), round(j.max(), 3)))
    if rels:
        L.append("\n**Fair decode-throughput ratio vs fp16_eager (same backend):**\n")
        L.append("| method | mean × | min × | max × |")
        L.append("| --- | --- | --- | --- |")
        for m, mn, lo, hi in rels:
            L.append(f"| {m} | {mn} | {lo} | {hi} |")
        L.append("\nAgainst the **same eager backend**, TurboQuant's per-token "
                 "decode is within this factor of fp16_eager — the gap vs "
                 "fp16_sdpa is the SDPA-kernel speedup, not an algorithmic cost "
                 "of TurboQuant.\n")

    # C. prototype blocker (from slow_micro, not run here)
    L.append("## C. Prototype python-loop blocker — BaseQuant / CARE-KV "
             "(slow_micro only)\n")
    sm = os.path.join(os.path.dirname(out_dir.rstrip("/")), "slow_micro",
                      "prefill_decode_all_rows.csv")
    if os.path.exists(sm):
        smdf = pd.read_csv(sm)
        smok = smdf[(smdf["status"] == "ok") &
                    (smdf["method"].isin(["basequant_int3", "basequant_int4",
                                          "adaptive_carekv_int3"]))].copy()
        for c in ["prefill_ms_mean", "decode_ms_per_token_mean"]:
            smok[c] = pd.to_numeric(smok[c], errors="coerce")
        L.append(tbl(smok[["method", "seq_len", "gen_tokens", "prefill_ms_mean",
                           "decode_ms_per_token_mean", "runtime_status"]]))
        L.append("\nThese are a **prototype Python-loop runtime blocker** "
                 "(`python_loop_runtime_blocker`), NOT an optimized algorithmic "
                 "runtime, and are **not** compared against the SDPA/eager rows.\n")
    else:
        L.append("_slow_micro results not found._\n")

    # 7. SDPA-after-reconstruct investigation
    L.append("## TurboQuant + SDPA investigation (requirement 7)\n")
    L.append("Can TurboQuant use SDPA after reconstructing K/V? **Implemented "
             "behind a flag, default `eager`; usable but NOT bit-equivalent.**")
    L.append("- The base path reconstructs full `K̂`/`V̂` (random-rotation + "
             "per-coordinate quant → dequant), so SDPA can run directly on "
             "`K̂`/`V̂`. ✅ (correct algebra)")
    L.append("- The **QJL residual correction is a pre-softmax logit term** "
             "(`attn_weights += qjl_corr`). SDPA does not expose logits, but its "
             "additive `attn_mask` plays the same role, so QJL folds in as "
             "`causal + qjl_corr/√d` — algebraically correct. ✅")
    L.append("- **But not bit-equivalent:** eager softmaxes in **fp32** while "
             "SDPA softmaxes in fp16; INT3 quantization widens the pre-softmax "
             "score range, amplifying that to ~1.5–3.9 abs logit (~16–20% rel) "
             "on a probe — even with SDPA's fp32 math backend. The **next-token "
             "argmax still matches**, so it is a usable approximate/faster "
             "backend, not an exact replacement.")
    L.append("- **Speed caveat:** a full `(B,Hq,Tq,Tk)` QJL bias forces SDPA's "
             "*math* backend, so the FlashAttention speedup only materializes "
             "with QJL disabled.")
    L.append("- Implemented as `TURBOQUANT_ATTENTION_BACKEND={eager,"
             "sdpa_reconstruct}` (**default `eager`**, unchanged — we do NOT "
             "force it). Probe details: "
             "`backend_control/turboquant_sdpa_equivalence.txt`.\n")

    L.append("## Honesty / preservation\n")
    L.append("- We do **not** claim TurboQuant is inherently slower than FP16: "
             "the fair comparator is fp16_eager (same backend); fp16_sdpa is the "
             "separate optimized ceiling.")
    L.append("- Backend is recorded per row (`backend_group`, "
             "`fair_runtime_group`, `attn_impl`).")
    L.append("- This control run does not modify or rerun fast_full; failed/"
             "unsupported rows are preserved.")
    rp = os.path.join(out_dir, "BACKEND_CONTROL_REPORT.md")
    with open(rp, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"[backend_control] wrote {rp}", flush=True)


def write_blocker_report(out_dir):
    """Write the runtime-blocker report to results/prefill_decode_perf/. Summarizes
    fast_full (measured) vs slow_micro (python_loop_runtime_blocker) vs guard-skipped
    rows, reading whatever all_rows CSV exists in out_dir. Never deletes anything."""
    import csv as _csv
    parent = os.path.dirname(out_dir.rstrip("/")) or out_dir
    report = os.path.join("results/prefill_decode_perf", "PREFILL_DECODE_RUNTIME_BLOCKER_REPORT.md")
    os.makedirs(os.path.dirname(report), exist_ok=True)
    # collect rows from both scope dirs if present
    rows = []
    for d in (os.path.join("results/prefill_decode_perf", "fast_full"),
              os.path.join("results/prefill_decode_perf", "slow_micro"), out_dir):
        p = os.path.join(d, "prefill_decode_all_rows.csv")
        if os.path.exists(p):
            try: rows += list(_csv.DictReader(open(p)))
            except Exception: pass
    # dedup
    seen = set(); uniq = []
    for r in rows:
        k = (r.get("model_id"), r.get("method"), r.get("seq_len"), r.get("batch_size"),
             r.get("gen_tokens"), r.get("runtime_scope"), r.get("status"))
        if k in seen: continue
        seen.add(k); uniq.append(r)
    rows = uniq
    from collections import Counter
    by_status = Counter(r.get("status", "") for r in rows)
    by_scope = Counter(r.get("runtime_scope", "") for r in rows)
    skipped = [r for r in rows if r.get("status") == "skipped_prototype_runtime_infeasible"]
    micro = [r for r in rows if r.get("runtime_status") == "python_loop_runtime_blocker"]
    L = ["# Prefill/Decode runtime BLOCKER report\n",
         "> **Runtime honesty note.** The patched BaseQuant / Adaptive-CARE-KV decode path is a "
         "per-(layer, kv_head, token) **Python-loop prototype** (~thousands of ms/token), not an "
         "optimized algorithmic runtime. fp16 and TurboQuant-style run at normal speed. Therefore the "
         "slow methods are measured **micro-only** (or skipped by the guard) and must NOT be read as "
         "achievable CARE-KV runtime. This report exists so the slow numbers are never mistaken for the "
         "method's algorithmic cost.\n",
         "## Runtime scopes\n",
         "- **fast_full** (measured, full grid): fp16, turboquant_int3, turboquant_int4.\n",
         "- **slow_micro / prototype_micro_only** (`python_loop_runtime_blocker`): basequant_int3, "
         "basequant_int4, adaptive_carekv_int3 — micro config only.\n",
         "## Row counts (kept; nothing deleted)\n",
         "| key | count |", "|---|---|"]
    for k, v in by_status.most_common():
        L.append(f"| status=`{k}` | {v} |")
    for k, v in by_scope.most_common():
        if k: L.append(f"| runtime_scope=`{k}` | {v} |")
    L.append(f"\n- guard-skipped rows (`skipped_prototype_runtime_infeasible` / "
             f"`python_loop_decode_path_too_slow`): **{len(skipped)}**")
    L.append(f"- micro-only prototype-blocker rows: **{len(micro)}**\n")
    L.append("## Slow-method guard\n")
    L.append("`--max-decode-ms-per-token <ms> --prototype-slow-method-policy {run,skip,micro_only}`: once a "
             "method's measured decode_ms_per_token exceeds the threshold (or `micro_only` after the first "
             "config), its later rows are recorded `skipped_prototype_runtime_infeasible` with "
             "`failure_reason=python_loop_decode_path_too_slow` instead of being run.\n")
    L.append("> " + SLOW_MICRO_NOTE + "\n")
    open(report, "w").write("\n".join(L) + "\n")
    print(f"[blocker] wrote {report} ({len(rows)} rows; skipped={len(skipped)}, micro={len(micro)})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/prefill_decode_perf")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--methods", default=",".join(METHODS.keys()))
    ap.add_argument("--gpu", default=None, help="CUDA device index to pin to")
    ap.add_argument("--preset", choices=["fast_full", "slow_micro", "tiered",
                                         "backend_control"], default="tiered")
    ap.add_argument("--smoke", action="store_true",
                    help="smoke: Mistral SL128 B1 gen16 all methods, 1 warmup/2 reps")
    ap.add_argument("--max-decode-ms-per-token", type=float, default=None,
                    help="slow-method guard threshold; once a method exceeds it, later "
                         "rows for that method are skipped_prototype_runtime_infeasible")
    ap.add_argument("--prototype-slow-method-policy", choices=["run", "skip", "micro_only"],
                    default="run", help="run=measure all; skip=skip a method's later rows "
                    "once it trips the threshold; micro_only=run one micro config per method then skip")
    args = ap.parse_args()

    # GPU pinning awareness: respect an externally-set CUDA_VISIBLE_DEVICES, or pin
    # to --gpu, and print the selected device so the run is unambiguous.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    _cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "(all)")
    try:
        _dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception:
        _dev = "?"
    print(f"[gpu] CUDA_VISIBLE_DEVICES={_cvd}  selected device 0 = {_dev}  "
          f"preset={args.preset}  guard(max_ms={args.max_decode_ms_per_token}, "
          f"policy={args.prototype_slow_method_policy})", flush=True)
    guard = dict(max_ms=args.max_decode_ms_per_token, policy=args.prototype_slow_method_policy)

    if args.smoke:
        sg = {m: dict(seqlens=[128], batches=[1], gens=[16], n_warmup=1, n_reps=2,
                      scope="smoke") for m in METHODS}
        run(["mistralai/Mistral-7B-v0.3"], sg, args.out_dir, smoke_tag="_smoke", guard=guard)
        write_blocker_report(args.out_dir)
        return

    if args.preset == "fast_full":
        # fp16 + TurboQuant INT3/INT4, full grid, reps5/warmup2
        os.makedirs(args.out_dir, exist_ok=True)
        models = MODELS
        grids = {m: dict(seqlens=[128, 256, 512, 1024], batches=[1, 2, 4],
                         gens=[32, 128], n_warmup=2, n_reps=5,
                         scope="fast_full(seq128-1024,batch1-4,gen32/128,reps5)",
                         runtime_scope="fast_full", runtime_status="measured")
                 for m in ("fp16", "turboquant_int3", "turboquant_int4")}
        run(models, grids, args.out_dir, guard=guard)
        write_blocker_report(args.out_dir)
        return

    if args.preset == "backend_control":
        # Backend-fairness control: SDPA-optimized fp16 baseline + same-backend
        # eager comparison (fp16_eager vs TurboQuant eager). Small grid, NOT a
        # rerun of fast_full. Mistral-7B + Yi-6B.
        os.makedirs(args.out_dir, exist_ok=True)
        models = ["mistralai/Mistral-7B-v0.3", "01-ai/Yi-6B"]
        grids = {m: dict(seqlens=[128, 512, 1024], batches=[1, 4], gens=[32, 128],
                         n_warmup=2, n_reps=5, scope="backend_control",
                         runtime_scope="backend_control", runtime_status="measured")
                 for m in ("fp16_sdpa", "fp16_eager", "turboquant_int3", "turboquant_int4")}
        run(models, grids, args.out_dir, guard=guard)
        write_backend_control_report(args.out_dir)
        return

    if args.preset == "slow_micro":
        # BaseQuant INT3/INT4 + CARE-KV INT3, micro diagnostic only:
        # Mistral-7B, SL128, B1, gen4, reps1, warmup1. Labelled as a Python-loop
        # runtime blocker — NOT an optimized algorithmic runtime.
        grids = {m: dict(seqlens=[128], batches=[1], gens=[4], n_warmup=1, n_reps=1,
                         scope="slow_micro(SL128,B1,gen4,reps1) prototype_micro_only",
                         runtime_scope="prototype_micro_only",
                         runtime_status="python_loop_runtime_blocker",
                         notes_override=SLOW_MICRO_NOTE)
                 for m in ("basequant_int3", "basequant_int4", "adaptive_carekv_int3")}
        os.makedirs(args.out_dir, exist_ok=True)
        run(["mistralai/Mistral-7B-v0.3"], grids, args.out_dir, guard=guard)
        write_blocker_report(args.out_dir)
        return

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    grids = build_tiered_grids()
    sel = set(args.methods.split(","))
    grids = {k: v for k, v in grids.items() if k in sel}
    run(models, grids, args.out_dir, guard=guard)
    write_blocker_report(args.out_dir)


if __name__ == "__main__":
    main()
