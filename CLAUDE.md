# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

CARE-KV: a research implementation of low-bit (INT2/INT3/INT4) KV-cache
quantization for LLaMA-family models, augmented with **sparse residual
correction** chosen by an output-error-aware router. Targets HuggingFace
`LlamaForCausalLM` via monkey-patching.

## Import / path convention

Package imports as **`CARE_KV.care_kv`**, not `care_kv`. Repo lives at
`~/CARE_KV/care_kv/`. All scripts assume `PYTHONPATH=/home/soeun` (the
grandparent of this directory). Always:

```bash
cd /home/soeun/CARE_KV/care_kv
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun
```

---

## 1. Current project status — method-complete

Stable and validated:

- **Post-RoPE K storage** in the cache (decode reads RoPE-correct K).
- **GQA-aware KV-head-indexed cache layout** (no per-query-head duplication).
- `CAREKV_PREFILL_MODE=carekv_stored` reads **only real stored residual slots**
  (no oracle leak, no upper-bound shortcut).
- `use_cache=True` / HF `DynamicCache` incremental decode works end-to-end
  (open-page append, no fresh page per token).
- **Packed INT2/INT3/INT4 base KV** storage (`CAREKV_PACKED_BASE=1`).
- **`scale_quant=int8`** with per-page master scale.
- **Absolute K/V budgets** (`STORE_ABS_K/V`, `READ_ABS_K/V`) — the current
  paper path; ratio budgets are legacy.
- **Score-normalized `joint` routing** is the current best policy.
- `correction_impl=cached` (pre-unpacked-slot cache per kv_head) is the
  **main stable runtime path**.
- `correction_impl=vectorized` exists and matches cached within fp16 noise
  for V-only, but **joint+both falls back to cached for bit-exactness**.
- `R=0 / READ_BUDGET=0 ≡ base_quant` invariant is exact (pytest-locked).

---

## 2. Paper-best configuration (locked)

Do not silently change these. If you ablate, gate behind a flag and keep
the defaults below.

```
BASE_BITS=3
CAREKV_PACKED_BASE=1
CAREKV_SCALE_QUANT=int8
CAREKV_PREFILL_MODE=carekv_stored
CAREKV_PREFILL_RESIDUAL_KIND=both
CAREKV_ROUTE_POLICY=joint
CAREKV_SCORE_NORMALIZE=1
CAREKV_CORRECTION_IMPL=vectorized   # was cached; combined selector is vectorized-only (§10)
CAREKV_K_CORRECTION_MODE=exact      # exact softmax renorm, not 1st-order Jacobian (§10)
CAREKV_KSCORE_LIVE=1                # combined_kvscore selector (§5g, §10)
CAREKV_BUDGET_POLICY=uniform
STORE_ABS_K=2
STORE_ABS_V=4
READ_ABS_K=2
READ_ABS_V=2
```

**Promoted 2026-07-15** from `linear` / `cached` / current-selector to
`exact` + `combined` (both levers) on the strength of the §10 NS=32 sweep
(7/7 over TurboQuant) confirmed by the DeepSeek NS=64 recheck
(`combined_exact` 9.3351 beats Turbo 9.5899 by −0.255, reproducing the NS=32
−0.241; the two-lever stack clears Turbo where neither lever alone does). Gates
passed: READ=0 ≡ base_quant bit-exact under both levers; DeepSeek NS=64 win
holds. Three coupled changes, not independent knobs:
- `K_CORRECTION_MODE=exact` — replaces the divergent 1st-order ΔO_K; also
  removes the `k_corr_scale=0.1` mis-scaling (§10).
- `KSCORE_LIVE=1` — the `combined_kvscore` selector; super-additive with
  `exact` on the hard-outlier tail (DeepSeek/Llama-2-13B flip only when both
  are on).
- `CORRECTION_IMPL=cached → vectorized` — **required**: the cached router has
  no KSCORE handling, so `combined` silently falls back to the `current`
  selector under `cached` (§10 gate). `vectorized` is faithful to `cached` for
  the non-KSCORE config (Δ=1.79e-07, §5g) and faster.

The prior `linear` / `cached` / current-selector path is still valid and
flag-reachable (set `CAREKV_K_CORRECTION_MODE=linear`, unset
`CAREKV_KSCORE_LIVE`, `CAREKV_CORRECTION_IMPL=cached`) — keep it for
reproducing pre-promotion results and for the READ=0 cached invariant test.

---

## 3. Main result summary

Synthetic sanity (SEQ_LEN=64, TinyLlama-1.1B):

- base_quant INT3 PPL **≈ 4.2831**
- optimized CARE-KV PPL **≈ 3.8294**
- improvement **≈ 10.6%**

Memory (`memory_table.csv`):

- packed_base + scale_quant=int8 reaches **≈ 0.24–0.26× FP16 KV** at
  context ≥ 2k.

Decode:

- `use_cache=True` incremental decode works.
- Latency is **prototype-level**: per-(layer, kv_head, t) Python loops in
  prefill + cached correction; HF DynamicCache dummy fp16 K/V inflates
  peak GPU memory. Both are documented in `summaries/remaining_improvements.md`.

WikiText-2 paper run (TinyLlama, SL=128, N=16): see
`results/paper_eval_20260529_015053/final_report.md` §9.

---

## 4. Historical pitfalls — do not repeat

- **Old zero-read `carekv_stored` rows are invalid.** Any row with
  `K_reads=0` AND `V_reads=0` under `carekv_stored` means the router
  never fired (typically because `STORE_BUDGET_RATIO=0` AND no absolute
  budget was set). Treat as superseded.
- **`carekv_eval` is an upper-bound diagnostic, not the paper method.**
  Never present `carekv_eval` numbers as final results.
- **The paper method is `carekv_stored`.**
- **Ratio budgets saturate.** `STORE_BUDGET_RATIO` / `READ_BUDGET_RATIO`
  hit a flat region quickly; use absolute budgets (`STORE_ABS_K/V`,
  `READ_ABS_K/V`) for the paper path.
- **`u_shaped` / `sensitivity` layer-budget policies did NOT improve over
  `uniform`** at the optimized absolute-budget level. Keep `uniform`.
- **K-only was best _only_ before absolute-budget tuning.** Current best
  is `kind=both` + `route_policy=joint` + `score_normalize=1`.
- **READ=0 invariant must hold.** `STORE_*=0` and `READ_*=0` (or
  `RESIDUAL_RATIO=0 + MIN_RESIDUALS=0`) must produce output bit-identical
  to `base_quant`. This is the invariant gating safe refactors.
- **Always check `K_reads`/`V_reads` for `carekv_stored` results.**
  Zero reads = router didn't fire = invalid CARE-KV claim.

---

## 5. Paper-eval directory structure

```
results/paper_eval_20260529_015053/
├── README.md                ← reproduce guide + headline numbers
├── final_report.md          ← auto-generated single-page summary
├── artifact_list.txt
├── summaries/               ← hand-curated markdown tables
├── figures/                 ← per-layer PNGs + 8 paper summary PNGs (fig_*.png)
├── ppl/                     ← synthetic-prompt PPL (sanity)
├── ppl_dataset/             ← WikiText-2 paper N=16 + multi-model N=4
├── sweeps/                  ← budget / Pareto sweeps
├── ablations/               ← V/K/Both, route policy, layer budget
├── memory/                  ← measured KV memory vs estimator vs fp16
├── latency/                 ← prefill + decode + vectorization bench
├── long_context/            ← Phase J synthetic retrieval/copy
├── generation/              ← qualitative samples
└── logs/                    ← gitignored
```

---

## 6. Main scripts / tools

End-to-end:

- `scripts/run_all_paper_eval.sh` — full paper-matrix runner (~4 h on one
  A100/H100). Honors env knobs `WT2_NUM_SAMPLES`, `LONG_CTX_TRIALS`,
  `SKIP_HEAVY=1` for fast smoke.

Per-section eval drivers:

- `scripts/run_wikitext2_ppl.sh`           — WT-2 PPL on the paper-best config
- `scripts/run_multimodel_ppl_eval.sh`     — WT-2 across a `$MODELS` list
- `scripts/run_long_context_retrieval.py`  — Phase J synthetic tasks
- `scripts/debug_stored_slot_reads.sh`     — sanity check the router is firing
- `tools/paper_eval.py {memory,figures,...}` — multi-subcommand driver
- `tools/eval_prefill_vectorization.py`    — Phase P4 vectorized-V bench
- `tools/eval_layer_budget_policies.py`    — Phase E layer budget ablation
- `eval_ppl_dataset.py`                    — HF-datasets PPL driver (wikitext/c4)

Wrap-up:

- `tools/summarize_all_results.py <paper_dir>` — regenerate `final_report.md`
  + `artifact_list.txt` from the CSVs.
- `tools/make_paper_figures.py --paper-dir <paper_dir>` — generate the 8
  paper summary PNGs.

---

## 7. What to run first in a new session

Safe order — confirms env, source builds, tests pass, and the existing
paper artifacts are still intact:

```bash
cd /home/soeun/CARE_KV/care_kv
source /home/soeun/anaconda3/etc/profile.d/conda.sh
conda activate vllm-carekv
export PYTHONPATH=/home/soeun

# 1. Source compiles
python -m py_compile quantizer.py cache.py residual_store.py \
    residual_router.py attention.py layer.py llama_patch.py utils.py \
    __init__.py tools/*.py

# 2. Tests pass (18 tests, ~70 s)
python -m pytest -q tests/test_carekv_v2.py

# 3. Regenerate the report from existing CSVs (no experiment re-run, ~1 s)
python tools/summarize_all_results.py results/paper_eval_20260529_015053
```

For a fast smoke of the full pipeline (skip heavy carekv cells, ~15 min):
```bash
PAPER_DIR=results/paper_eval_20260529_015053 SKIP_HEAVY=1 \
  bash scripts/run_all_paper_eval.sh
```

---

## 8. Remaining work

Engineering:

- More robust WikiText-2 / C4 paper-scale runs (SL ≥ 512, N ≥ 32) once
  joint+both is vectorized.
- Multi-model coverage beyond TinyLlama + JackFram/llama-160m (needs HF
  auth or larger local cache).
- Stronger long-context benchmark (RULER / LongBench) or a larger model
  where fp16 actually solves the task at scale.
- **Vectorized joint+both correction** — top-priority unblocker for
  large-SL paper evals.
- **Custom lightweight HF Cache** to avoid the DynamicCache dummy fp16
  K/V we currently feed `get_seq_length()` (would cut peak GPU mem ~50%).
- CUDA / Triton kernels for packed unpack + correction.

Paper:

- Final paper writing.

---

## 9. Rules for future work

1. **Do not silently change the paper-best configuration** (§2).
2. **Keep old behavior behind flags.** Don't rip out `carekv_eval`,
   ratio-budget paths, or non-packed base — they're still useful for
   ablation and would break old result reproduction if removed.
3. **Label results clearly** in every report:
   - `sanity / smoke` — short synthetic prompt, N ≤ 4.
   - `diagnostic` — per-layer or debugging artifact.
   - `prototype latency` — current Python-loop wall-clock, not the
     achievable runtime.
   - `paper-ready` — passes the invariant + config gates and is in
     `final_report.md`.
4. **Never present `carekv_eval` as the final method.** Paper method is
   `carekv_stored`.
5. **Never present zero-read `carekv_stored` rows as final.** Always
   check `K_reads + V_reads > 0` for any CARE-KV claim.
6. **Always check `K_reads` / `V_reads`** for `carekv_stored` cells
   before believing the PPL.
7. **Preserve the `READ=0 ≡ base_quant` invariant.** It is the gate for
   safe refactors of the router and correction paths.
8. **Preserve both `USE_CACHE=0` and `USE_CACHE=1` paths.** The
   eval-script path uses `USE_CACHE=0` (full prefill); the streaming
   generation path uses `USE_CACHE=1` (DynamicCache + open-page append).
   Don't conflate or remove either.

---

## 10. Exact K correction (`CAREKV_K_CORRECTION_MODE=exact`) — screening GO

The default ΔO_K is the **1st-order Jacobian** of softmax w.r.t. the key
residual. Writing `δs_t = (q·R_K,t)/√D` for the logit perturbation carried by
the *selected* K slots, that Jacobian is the `δs→0` limit of

```
a_new = softmax(s_base + δs) = a_base·e^δs / Σ_u a_base,u·e^δs,u
O_new = Σ_t a_new,t · (V_base,t + R_V,t·[t selected])
```

which needs **the same slot reads** — one extra `exp` + one matmul over (Q, N).
`exact` computes that directly (`attention.py:exact_softmax_correction`).

Two bugs the linear form was hiding:

1. **The apply path omits `1/√D`.** `attention.py`'s cached + vectorized K
   apply uses the raw `q·R_K`, not `(q·R_K)/√D` — while `s_base` *is* scaled by
   `1/√D`. `CAREKV_K_CORRECTION_SCALE=0.1` is an empirical stand-in for the
   missing `1/√(head_dim)` (0.125 at D=64, 0.088 at D=128), so it is
   **mis-scaled per model**. (`layer.py`'s `python`/`carekv_eval` path *does*
   apply `scale_val`, so the three impls never agreed on this term.)
2. **The linearization diverges.** `exp` is convex, so extrapolating off its
   tangent overshoots once `|δs| ≳ 1` — exactly the outlier-heavy-K regime
   where CARE-KV loses to rotation baselines. `a_new` is a softmax, so `exact`
   is bounded by construction (`‖O_new‖ ≤ max‖V‖`) and needs no `k_corr_scale`
   damping at all. This is why the `clamp`/`nguard` band-aids in
   `results/kstab_screening/` failed: they bound `δs` crudely instead of
   fixing the estimator.

Invariants (`tests/test_exact_k_correction.py`, 37 checks):

- READ=0 ≡ base_quant stays **bit-exact** (Δ=0.0) — falls through when no K
  slot is read, so the §9.7 refactor gate is untouched.
- `kind="v"` is bit-identical to `linear` (δs ≡ 0).
- cached == vectorized under exact (Δ≤1.2e-07).
- exact == brute-force softmax; bounded at δs~64 where linear overshoots 17×.

Screening (WT-2, SL512, **NS=8 — screening scale, not paper-ready**):

| model | outlier (base−turbo) | turbo | linear | exact | Δexact | exact vs turbo |
|---|---:|---:|---:|---:|---:|---|
| Mistral-7B-v0.3 | −0.090 | 8.187 | 7.961 | **7.938** | −0.023 | −0.249 win |
| Yi-6B | 0.514 | 8.844 | 8.969 | **8.819** | −0.150 | −0.025 **lose→win** |
| DeepSeek-7B | 0.752 | 9.957 | 10.217 | **10.130** | −0.087 | +0.173 lose |
| Llama-2-13B | 0.506 | 7.067 | 7.318 | **7.299** | −0.019 | +0.232 lose |

- **exact improves 4/4 and regresses none, at identical reads and runtime**
  (DeepSeek 2460 s → 2431 s). It is a free strict improvement.
- It **flips Yi-6B** from a Turbo loss to a Turbo win, and narrows DeepSeek's
  gap by 33%.
- **The "gain is monotone in K-outlier severity" hypothesis did NOT hold**:
  Llama-2-13B (outlier 0.506) gains least (−0.019). Do not claim it. Note the
  outlier proxy itself is NS-unstable — at NS=8 Turbo *loses* to base on
  Mistral (8.187 vs 8.097) but wins at NS=32 in §5g.
- Yi's winning margin (−0.025) is inside NS=8 noise. **NS=32 confirmation is
  the gate** before promoting `exact` to the paper-best default.

### NS=32 confirmation — COMPLETE (7 models, all §5g Llama-arch models)

Reference arms (fp16 / base / turbo / linear) reproduce §5g to 4 decimals (same
`run_one` windowing), so these rows drop straight into the §5g table. Sorted by
K-outlier severity (`base − turbo`, ascending). `linear` = `carekv_current`
(current selector); `exact` = same selector + `CAREKV_K_CORRECTION_MODE=exact`.

| model | outlier | fp16 | base | turbo | linear | exact | Δexact | exact vs turbo |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mistral-7B-v0.3 | 0.044 | 6.7697 | 7.2987 | 7.2545 | 7.1152 | **7.0986** | −0.0166 | −0.156 win |
| SOLAR-10.7B | 0.096 | 6.0587 | 6.4307 | 6.3350 | 6.2937 | **6.2856** | −0.0081 | −0.049 win |
| OpenLLaMA-7B-v2 | 0.255 | 8.0538 | 8.7886 | 8.5333 | 8.5560 | **8.5372** | −0.0188 | +0.004 ~tie |
| Llama-2-13B | 0.485 | 6.1465 | 6.8961 | 6.4111 | 6.6549 | **6.6070** | −0.0479 | +0.196 lose |
| Yi-6B | 0.519 | 7.2063 | 8.3244 | 7.8057 | 7.8678 | **7.7135** | −0.1543 | −0.092 win |
| DeepSeek-7B | 0.748 | 8.4432 | 9.7294 | 8.9816 | 9.2008 | **9.1402** | −0.0606 | +0.159 lose |
| TinyLlama-1.1B | 1.590 | 9.9918 | 14.0372 | 12.4473 | 11.9229 | **11.6825** | −0.2404 | −0.765 win |

**NS=32 verdict — exact improves 7/7, regresses 0, flips 2 vs Turbo:**

- **Free strict improvement on every model** (Δexact < 0 for all 7), at
  identical K/V reads and identical runtime. This is the headline.
- **exact + the plain `current` selector matches what §5g needed `combined`
  for, and adds Yi.** §5g `current`-vs-Turbo won 3/7 (Mistral, SOLAR,
  TinyLlama); `current+exact` wins **4** and ties a 5th — flipping **Yi-6B
  lose→win** (7.868→7.7135, and the margin *grows* NS=8→32: −0.025→−0.092) and
  **OpenLLaMA-7B lose→tie** (8.556→8.5372 vs turbo 8.5333, +0.004). §5g reached
  a 4-win record only with the `combined` selector; `exact` reaches it with
  `current` alone, so **`combined`+`exact` stacked is an untested upside**
  (the two levers are orthogonal — selection vs. estimator).
- **DeepSeek-7B / Llama-2-13B improve but do NOT flip.** Turbo gap narrows 28%
  on DeepSeek (+0.219→+0.159) and 20% on Llama-2-13B (+0.244→+0.196). The two
  heaviest-outlier models stay rotation's territory — `exact` fixes the
  *estimator*, not the *un-rotated base* (expected §6 limit: once K-outliers
  are severe the base quantizer has already discarded what no residual can
  recover).
- **The "gain ∝ outlier severity" hypothesis is dead** (7-model confirmation):
  ordering by outlier, Δexact is non-monotone — Yi (0.519) gains −0.154 but
  higher-outlier DeepSeek (0.748) gains only −0.061; Llama-2-13B (0.485) gains
  least among mid-outlier models (−0.048). The gain tracks *how much the
  linearization was overshooting* (a per-token δs-magnitude property), not the
  base's outlier severity. TinyLlama's −0.240 is the small-model regime
  (correction ≫ rotation), not an outlier effect. Do not conflate the axes.

**Score (current+exact vs Turbo, 7 models):** 4 WIN (Mistral, SOLAR, Yi,
TinyLlama), 1 tie (OpenLLaMA-7B), 2 lose (Llama-2-13B, DeepSeek-7B). Same 4-1-2
record §5g reached with `combined`, but reached with `current` alone and with a
different, arguably stronger win set (Yi is a clean win here vs a tie in §5g).

### combined + exact STACKED (both levers) — 7/7 beat Turbo, NS=32

The selector (`combined`, `CAREKV_KSCORE_LIVE=1`) and the estimator (`exact`)
are orthogonal levers. Stacking both — `combined_exact` — **beats TurboQuant on
all 7 models**, including the two heaviest-outlier models (DeepSeek, Llama-2-13B)
that **neither lever alone could flip**. `Δstack` = `combined_exact − combined`
(exact's gain on top of the combined selector). All arms share identical budgets
and same `run_one` windowing (refs reproduce §5g to 4 dp).

| model | outlier | turbo | cur | cur+ex | comb | **comb+ex** | Δstack | comb+ex vs turbo |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Mistral-7B | 0.044 | 7.255 | 7.115 | 7.099 | 7.011 | **6.919** | −0.092 | −0.336 win |
| SOLAR-10.7B | 0.096 | 6.335 | 6.294 | 6.286 | 6.301 | **6.160** | −0.141 | −0.175 win |
| OpenLLaMA-7B | 0.255 | 8.533 | 8.556 | 8.537 | 8.291 | **8.220** | −0.071 | −0.314 win |
| Llama-2-13B | 0.485 | 6.411 | 6.655 | 6.607 | 6.500 | **6.342** | −0.158 | **−0.069 win** |
| Yi-6B | 0.519 | 7.806 | 7.868 | 7.713 | 7.811 | **7.460** | −0.351 | −0.346 win |
| DeepSeek-7B | 0.748 | 8.982 | 9.201 | 9.140 | 9.237 | **8.741** | −0.496 | **−0.241 win** |
| TinyLlama-1.1B | 1.590 | 12.447 | 11.923 | 11.682 | 10.918 | **10.451** | −0.468 | −1.997 win |

- **7/7 vs Turbo — the first configuration that beats rotation everywhere in
  this cohort.** The two severe-outlier models rotation used to own now flip:
  DeepSeek 9.201/9.140/9.237 (all lose) → **8.741 win**; Llama-2-13B
  6.607/6.500 (lose) → **6.342 win**.
- **The two levers are super-additive on the hard tail, not just additive.** On
  DeepSeek, `combined` alone *hurts* (9.237 > current 9.201) and `exact` alone
  only reaches 9.140 (still loses) — yet stacked they hit 8.741. Neither lever
  crosses the Turbo line alone; together they clear it by −0.241.
- **exact helps the `combined` selector MORE than the `current` selector**
  (Δstack ≫ current→cur+ex): DeepSeek −0.496 vs −0.061, Yi −0.351 vs −0.154,
  TinyLlama −0.468 vs −0.240. **Mechanism only partly understood:** on Llama-2
  and Yi, `combined` shifts read budget toward K (K/V 0.84→1.28, 0.78→0.91),
  so the bounded renormalization has more/larger δs to fix — consistent. But
  **DeepSeek contradicts the simple story** (`combined` reads slightly *less* K,
  K/V 0.74→0.70, yet gets the biggest exact boost). Do not assert the K-budget
  explanation as general; the DeepSeek −0.496 cell is the most surprising and
  should get an **NS=64 recheck** before it goes in the paper.
- **Caveat:** `combined`/`KSCORE_LIVE` is a "net-positive but variable lever"
  per §5g (helps most, flat/slightly-negative on a few *alone*). Its value here
  is almost entirely realized *in combination with* `exact`, not standalone.

**Bottom line:** `CAREKV_K_CORRECTION_MODE=exact` + `CAREKV_KSCORE_LIVE=1` is the
strongest CARE-KV configuration measured — 7/7 over TurboQuant at NS=32, no
regression anywhere, identical reads/runtime to `linear`. Candidate for the new
paper-best (§2), pending the DeepSeek NS=64 recheck and a `READ=0` re-confirm
under `KSCORE_LIVE`.

**Gate results (2026-07-14):**
- **READ=0 invariant under `KSCORE_LIVE=1` + `exact`: PASS, bit-exact (Δ=0.0)**
  for kind ∈ {v,k,both}. The §9.7 refactor gate holds with both levers on.
- **`KSCORE_LIVE` is `vectorized`-only.** The cached path
  (`residual_router.py:route`) has NO KSCORE handling, so under `KSCORE_LIVE=1`
  cached and vectorized compute *different selectors* (cached falls back to the
  `current` proxy) and diverge (Δ≈0.5 linear, 0.12 exact at the unit fixture) —
  this is pre-existing selector wiring, NOT an `exact` defect (with KSCORE off,
  exact matches cached at 1e-7). **All combined/combined_exact PPLs above used
  `correction_impl=vectorized`, so they are self-consistent.** Consequence: if
  `combined_exact` becomes paper-best, §2 must also switch
  `CAREKV_CORRECTION_IMPL=cached → vectorized` (faithful per §5g, and faster);
  the `combined` selector cannot run on the cached path as written.
- DeepSeek NS=64 recheck of the −0.496 Δstack cell: running
  (`results/exact_kcorr/deepseek7b_ns64.csv`).

Cost: `exact` ignores `CAREKV_K_CORRECTION_SCALE` and makes
`CAREKV_K_QDOTR_CLAMP_PCT` / `CAREKV_K_NORM_GUARD_PCT` unnecessary.

Driver: `tools/eval_exact_kcorr.py` (same `run_one` windowing as
`tools/eval_combined_vs_turbo.py`, so refs compare directly to §5g).

---

## Runtime knobs cheat-sheet

| Env var | Values | Effect |
|---|---|---|
| `CAREKV_PREFILL_MODE` | `fp` / `base_quant` / `carekv_eval` / `carekv_stored` | Prefill path. **Use `carekv_stored` for paper.** |
| `CAREKV_PREFILL_RESIDUAL_KIND` | `v` / `k` / `both` | Which residual to apply. **`both` for paper.** |
| `CAREKV_ROUTE_POLICY` | `joint` / `separate` / `k_first` / `adaptive` | Routing policy. **`joint` for paper.** |
| `CAREKV_SCORE_NORMALIZE` | 0 / 1 | Per-kind normalization for `joint`. **1 for paper.** |
| `CAREKV_CORRECTION_IMPL` | `python` / `cached` / `vectorized` | Correction kernel. **`vectorized` for paper** (required for `KSCORE_LIVE`; was `cached` pre-2026-07-15). |
| `CAREKV_K_CORRECTION_MODE` | `linear` / `exact` | ΔO_K estimator. `linear` = 1st-order Jacobian (code default). `exact` = renormalized softmax. **`exact` for paper** (§10). |
| `CAREKV_KSCORE_LIVE` | 0 / 1 | `combined_kvscore` K+V selector. Code default 0. **1 for paper** (§5g, §10). Vectorized-only. |
| `CAREKV_BUDGET_POLICY` | `uniform` / `u_shaped` / `sensitivity` | Per-layer budget multiplier. **`uniform` for paper.** |
| `CAREKV_PACKED_BASE` | 0 / 1 | Real packed INT base storage. **1 for paper.** |
| `CAREKV_SCALE_QUANT` | `none` / `int8` | Per-page scale quantization. **`int8` for paper.** |
| `BASE_BITS` | 2 / 3 / 4 | Base KV bit-width. **3 for paper.** |
| `STORE_ABS_K`, `STORE_ABS_V` | int | Absolute per-page store budget (K, V). **2, 4 for paper.** |
| `READ_ABS_K`, `READ_ABS_V` | int | Absolute per-decode read budget (K, V). **2, 2 for paper.** |
| `CAREKV_DEBUG_STATS` | 0 / 1 | Emit `K_reads` / `V_reads` counters. **Always set 1 for CARE-KV runs** — needed to validate the router fired. |
| `MODEL_ID` | HF id | Defaults to `TinyLlama/TinyLlama-1.1B-Chat-v1.0`. |
| `SEQ_LEN`, `NUM_SAMPLES` | int | PPL window length / count. |
