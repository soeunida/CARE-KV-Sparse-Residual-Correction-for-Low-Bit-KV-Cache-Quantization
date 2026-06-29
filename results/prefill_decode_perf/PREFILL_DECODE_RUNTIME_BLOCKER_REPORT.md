# Prefill/Decode runtime BLOCKER report

> **Runtime honesty note.** The patched BaseQuant / Adaptive-CARE-KV decode path is a per-(layer, kv_head, token) **Python-loop prototype** (~thousands of ms/token), not an optimized algorithmic runtime. fp16 and TurboQuant-style run at normal speed. Therefore the slow methods are measured **micro-only** (or skipped by the guard) and must NOT be read as achievable CARE-KV runtime. This report exists so the slow numbers are never mistaken for the method's algorithmic cost.

## Runtime scopes

- **fast_full** (measured, full grid): fp16, turboquant_int3, turboquant_int4.

- **slow_micro / prototype_micro_only** (`python_loop_runtime_blocker`): basequant_int3, basequant_int4, adaptive_carekv_int3 — micro config only.

## Row counts (kept; nothing deleted)

| key | count |
|---|---|
| status=`ok` | 291 |
| status=`unsupported` | 4 |
| runtime_scope=`fast_full` | 288 |
| runtime_scope=`prototype_micro_only` | 3 |
| runtime_scope=`nan` | 1 |

- guard-skipped rows (`skipped_prototype_runtime_infeasible` / `python_loop_decode_path_too_slow`): **0**
- micro-only prototype-blocker rows: **3**

## Slow-method guard

`--max-decode-ms-per-token <ms> --prototype-slow-method-policy {run,skip,micro_only}`: once a method's measured decode_ms_per_token exceeds the threshold (or `micro_only` after the first config), its later rows are recorded `skipped_prototype_runtime_infeasible` with `failure_reason=python_loop_decode_path_too_slow` instead of being run.

> Current patched cache path is dominated by per-layer/per-head/per-token Python loops and should not be interpreted as optimized algorithmic runtime.

