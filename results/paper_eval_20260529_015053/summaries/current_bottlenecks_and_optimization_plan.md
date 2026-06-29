# CARE-KV current bottlenecks and optimization plan

Audit of the remaining engineering bottlenecks and evaluation gaps in the
CARE-KV implementation as of `master` = `6933210` (post-merge of all
Phase J/K/L/K-c/K-d/K-e work).

Each item is scored with:
- **Category**: method / runtime / memory / evaluation
- **Difficulty**: small (≤ 1 day) / medium (1–3 days) / large (week+)
- **Expected benefit**: scale and dimension of the win
- **Priority**: 1 (now) / 2 (next) / 3 (later) / DONE / DEFERRED

The list is ordered roughly by priority. Items marked DONE were addressed
in this session and reference their commit/branch.

---

## 1. Routing baseline ablation — DONE (this session)

| | |
|---|---|
| Category | evaluation |
| Files | `tools/eval_routing_baselines.py` (new), `tools/make_routing_baseline_figure.py` (new), `cache.py:CacheConfig.baseline_score`, `residual_router.py` |
| Evidence pre-audit | No published comparison vs random/magnitude/attention; could not refute "the score doesn't matter, only the budget does" |
| Paper impact | **High.** A reviewer's first question. |
| Difficulty | small |
| Benefit | Establishes the routing score is load-bearing. **CARE-KV beats random by 33 PPL, magnitude-only by 19, attention-only by 3.4 PPL**; within 0.19 PPL of the magnitude×attention oracle on the synthetic prompt. |
| Priority | **DONE** (branch `feat/routing-baseline-ablation`, synthetic SL=64 smoke complete) |
| Follow-up | Run on WikiText-2 N=4 SL=128 (`RUN_PHASE_M=1 M_DATASET=wikitext`, ~2 h) for a paper-headline number, not a smoke run. |

---

## 2. Vectorized joint+both prefill correction — Priority 1 (next)

| | |
|---|---|
| Category | runtime |
| Files | `attention.py:vectorized_v_correction`, `layer.py:_apply_sparse_prefill_correction_stored`, `residual_router.py` |
| Evidence | `final_report.md` §13 limitation 1; vectorized V matches cached within 0.002 PPL at SL=128 (`prefill_vectorization_bench.csv`), but `joint+both` falls back to cached for bit-exactness. WT-2 paper run on TinyLlama at SL=128 N=16 took **6 268 s** (~1.75 h) — the carekv cell, dominated by the per-(layer, kv_head, t) Python loop. |
| Paper impact | **Very high.** Unblocks SL=512 N=32 paper evals, multi-model runs, and any reviewer ask for "scale beyond TinyLlama-1.1B". Today the runtime envelope forces every paper number to SL ≤ 128. |
| Difficulty | medium — requires reworking the K-correction path (`ΔO_K ≈ Σ a · (q·R_K) · (V − O_base)`) to run as a single batched op over `(H_q, T)` and reconciling the `joint+normalize` selection order with the vectorized read. |
| Benefit | Expected **3–5× prefill speedup** (vectorized V already gives 1.41× at SL=128 alone, and K is the slower kind). Drops the WT-2 carekv cell from ~1.75 h to ~25–35 min, making N=16 SL=512 feasible. |
| Priority | **1 (next).** Highest impact-per-engineering-hour item. |

---

## 3. HF DynamicCache fp16 dummy KV memory — Priority 2

| | |
|---|---|
| Category | memory |
| Files | `llama_patch.py:CAREKVLlamaAttention.forward`, `cache.py` (the real CAREKV cache) |
| Evidence | `final_report.md` §13 limitation 3; `summaries/remaining_improvements.md` §4. We stuff dummy `(B, Hkv, T, D)` zeros into HF DynamicCache so `get_seq_length()` advances. At SL=128 paper run, peak GPU mem was **5 425 MB** for carekv_stored INT3 vs **2 262 MB** for fp16 — about 2.4× fp16, dominated by the dummy. |
| Paper impact | **Medium.** Memory table in `final_report.md` §3 currently shows the *cache* (packed INT3 + int8 scale) at **0.243× FP16** at 2 048 tokens, but the *peak runtime memory* is inflated by the dummy. Reviewers will notice the gap. |
| Difficulty | medium — write a minimal `CAREKVCache(transformers.cache_utils.Cache)` subclass that lies about its shape: stores `(B, Hkv, T, 1)` zero tensors instead of `(B, Hkv, T, D)`, overrides `get_seq_length`, `to_legacy_cache`, `update`. |
| Benefit | Expected **~50 % peak GPU-memory reduction** during incremental decode (cuts ~3 GB at SL=2k on TinyLlama). Matches the cache memory story to the runtime memory story. |
| Priority | **2.** Visible improvement, well-scoped. |

---

## 4. Read-time sparse routing visualization — Priority 2

| | |
|---|---|
| Category | evaluation |
| Files | none yet — would be `tools/make_read_routing_3d.py` (new) |
| Evidence | Existing 3D / heatmap diagnostics (`carekv_before_after_3d.md`, `before_after_3d_figures.md`) show the **store-time** selection (which residuals are kept at prefill). We do **not** yet show the **read-time** selection (which stored slots are read per decode step), so a reviewer can't see whether routing dynamics differ from storage. |
| Paper impact | Medium. Complements Phase K-e (store-side) with a per-decode-step view; useful for the "how does the router behave during generation?" question. |
| Difficulty | small — analogous to the existing Phase K-c/d/e plotters, just hooking into `ResidualRouter.route()` to log selected slot indices per decode step. |
| Benefit | One new diagnostic figure (e.g., per-layer scatter of which stored slots get read at each step, with a small annotation summarizing read-rate per slot). Helps explain the **K/V read balance** story already surfaced in the routing baseline ablation (random gives 55k K / 125k V; carekv gives 82k K / 99k V). |
| Priority | **2.** Helpful but not load-bearing for the headline. |

---

## 5. Vectorized INT3 unpack on the read path — Priority 3

| | |
|---|---|
| Category | runtime |
| Files | `quantizer.py:unpack_int3_2d`, `cache.py` (base read path), `attention.py` (cached-impl unpack) |
| Evidence | Phase P5 introduced vectorized INT3 unpack via `(word.unsqueeze(-1) >> arange(8)*3) & 0x07`. The read path uses it for the **base** dequantize. The **cached correction** path's slot unpack is already pre-unpacked (Phase P4-cached). Likely small remaining win. |
| Paper impact | Low. |
| Difficulty | small — would need profiling first to confirm there's any hot spot left. |
| Benefit | Estimated **5–10 %** decode speedup if any. Probably already optimal. |
| Priority | **3 (later).** Profile first; do not optimize speculatively. |

---

## 6. Multi-model coverage beyond TinyLlama — Priority 2

| | |
|---|---|
| Category | evaluation |
| Files | `scripts/run_multimodel_ppl_eval.sh` (already exists) |
| Evidence | Today's coverage: TinyLlama-1.1B-Chat + JackFram/llama-160m only. `final_report.md` §10 + `multimodel_wikitext2.csv`. Llama-3.2-1B is HF-gated, no auth in session; larger LLaMA-family models not cached locally. |
| Paper impact | **Medium-high.** Reviewer will ask "does this generalize beyond TinyLlama?". JackFram-160m is a sanity check, not a model people use. |
| Difficulty | small — IF HF auth is available; otherwise blocked on credentials/disk. |
| Benefit | Adds at least one production-grade model row (Llama-3.2-1B/3B). Probably most credible in combination with bottleneck #2 above (need vectorized joint+both to keep per-cell runtime tractable on a larger model). |
| Priority | **2.** Gated on user-provided HF auth + disk budget. |

---

## 7. CUDA / Triton kernels — DEFERRED

| | |
|---|---|
| Category | runtime |
| Evidence | `summaries/remaining_improvements.md`. The cached + vectorized PyTorch paths are still Python-loop bound at O(T² × H_q × L) for prefill correction; CUDA/Triton kernels for packed unpack + correction would close most of the remaining gap to fp16 latency. |
| Paper impact | Low for current paper (correctness is locked); high for follow-up "serving" paper. |
| Difficulty | large (week+). Requires Triton expertise and careful numerics review. |
| Benefit | Expected **10–100× decode speedup**, would make CARE-KV serving-grade. |
| Priority | **DEFERRED** — out of scope for the current paper; reasonable next-paper item. |

---

## 8. C4 dataset PPL — Priority 3

| | |
|---|---|
| Category | evaluation |
| Files | `eval_ppl_dataset.py` (already supports `--dataset c4` via `allenai/c4` streaming) |
| Evidence | Driver exists but never executed. `final_report.md` §9 has WikiText-2 only. |
| Paper impact | Low-medium. Strengthens generalization claim. |
| Difficulty | small. |
| Benefit | One more dataset row in §9. Mostly mechanical. |
| Priority | **3.** Run when adding multi-dataset table to the paper. |

---

## 9. Long-context benchmark — DEFERRED

| | |
|---|---|
| Category | evaluation |
| Evidence | `summaries/long_context_retrieval_table.md`. Phase J ran on TinyLlama-tractable synthetic retrieval; fp16 only solves at n_pairs ≤ 6, ctx=128 — too small to be a "long-context" claim. |
| Paper impact | Medium — currently labelled "deferred" in the report (`final_report.md` §11). |
| Difficulty | medium — RULER or LongBench on a larger model is the right replacement. Blocked on the same multi-model / runtime constraints as #2 + #6. |
| Benefit | A real long-context number for the paper. |
| Priority | **DEFERRED** until #2 ships (vectorized joint+both prefill) AND #6 lands (Llama-3.2-3B or larger). |

---

## Priority summary

| Priority | Item | Category |
|---|---|---|
| **DONE** | 1. Routing baseline ablation | evaluation |
| **1** | 2. Vectorized joint+both prefill | runtime |
| **2** | 3. HF DynamicCache fp16 memory leak | memory |
| **2** | 4. Read-time routing visualization | evaluation |
| **2** | 6. Multi-model coverage (gated on HF auth) | evaluation |
| **3** | 5. Vectorized INT3 unpack (profile first) | runtime |
| **3** | 8. C4 dataset PPL | evaluation |
| **DEFERRED** | 7. CUDA / Triton kernels | runtime |
| **DEFERRED** | 9. Long-context benchmark | evaluation |

Recommended next action: **item 2** (vectorized joint+both prefill). It is
the only remaining item that materially unblocks the rest of the paper
evaluation envelope.
