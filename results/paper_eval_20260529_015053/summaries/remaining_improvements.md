# Remaining method-level improvements — assessment

Reviewed against the current paper-best CARE-KV config (PPL **3.8294** on
synthetic, **13.54** on WikiText-2 INT3, ~0.25× FP16 memory).  Each idea
below is rated **paper-ready / experimental / future work**, with a brief
note on the implementation risk and expected return.

---

## 1. Soft K-guided V routing  (`CAREKV_COUPLED_ROUTING=1`)

**Concept**: After K slots are selected for a query, V residual candidates
whose token range overlaps a selected K page (or near-overlaps it in
attention) get an additive score bonus.  V routing therefore becomes
*conditioned* on the K side instead of being independent.

**Status**: **experimental / not implemented in this round**

**Risk / cost**:
- Requires a new joint-aware scoring pass — wasn't shippable in this
  optimization window without breaking the cached/vectorized split.
- The `k_first` policy already encodes a related idea (fill K budget first,
  then conditionally fill V above threshold).  In our SEQ_LEN=64 ablation
  `k_first` produced PPL 4.0731 vs joint's 3.8294 — *worse* than joint.

**Expected return**: low-to-moderate.  At our tested budgets the joint
policy already lets V outscore K when V matters; explicit coupling adds
complexity without obvious head-room.

**Recommendation**: future work, conditional on observing a regime where
joint+normalize underperforms a simple coupled heuristic.

---

## 2. Entropy-aware read budget  (`CAREKV_ENTROPY_READ=1`)

**Concept**: Per-query, compute the softmax entropy of `a_base`.  Queries
with high entropy (attention spread thin) get a slightly enlarged read
budget; queries with low entropy (peaked attention) get a smaller one.
The total expected read budget across the network is preserved.

**Status**: **experimental / not implemented**

**Risk / cost**:
- Cheap to implement (entropy is a one-line op on `a_base`).
- The Phase E `adaptive` route_policy attempted an entropy-based K/V split
  and produced PPL 4.15 vs separate's 3.87 — *worse* — because at our
  budgets every query is already near its useful-read ceiling.  A
  *per-query budget* (not just split) might behave differently, but the
  same "everything is near ceiling" objection applies.

**Expected return**: low at the current budget levels (RK=RV=2).  Might
matter more at *very* low budgets where some queries are simply unreadable
and trading their budget to others helps.

**Recommendation**: future work; would justify a dedicated budget-grid
ablation rather than a one-off knob.

---

## 3. Vectorized joint+both prefill  (extension of Phase 32 vectorized V)

**Concept**: The Phase 32 vectorized V path is bit-equal to cached under
`route_policy=separate`.  For the *paper-best* config (`joint` + `both`)
we currently fall back to the cached per-(h, t) path because vectorized V
doesn't reproduce the joint K/V interleaved top-k.  Implementing K
scoring + per-query normalized joint top-k as batched torch ops would
extend the 1.41× decode-speedup (from cached) to the paper config too.

**Status**: **future work, high-value**

**Risk / cost**:
- ~150-300 lines of careful tensor work: sketches + boundary_risk +
  V_diff vectorized, then normalized joint top-(budget_k+budget_v).
- PPL must match cached bit-equally (already verified V alone matches at
  SEQ_LEN=128 within 0.002 PPL).  Tie-breaking between `torch.topk` and
  Python `sorted()` on near-equal scores caused a 0.11 PPL drift at
  SEQ_LEN=64 — would need to be revisited with the K-side added.

**Expected return**: ~1.4× wall-clock on every carekv_stored eval.  At
WikiText-2 N=16 this means ~1 hour saved per run.  At paper-scale
(N=32 at SEQ_LEN=512) it means hours saved.

**Recommendation**: **highest-priority follow-up** — would unblock real
SEQ_LEN=512 paper-grade evals.

---

## 4. Custom lightweight HF Cache metadata

**Concept**: HF `DynamicCache` currently holds the full fp16 K/V we feed
it as dummy values to advance `get_seq_length()` (see § 0g).  At
TinyLlama T=512 that's ~5 GB of GPU peak we don't actually use for
attention.  A custom Cache subclass that stores ONLY (B, Hkv, T, 0) zero
tensors — or just an integer counter — would reduce peak by the same
~3 GB observed in Phase H's bench.

**Status**: **medium risk, paper-relevant — feasibility assessed below**

**Feasibility**:
- HF's `Cache` ABC requires `update()` returning `(key_concat,
  value_concat)`.  Some downstream code (sliding window attention, some
  generation utilities) inspects those tensors.  We can't easily return
  zero-shape tensors and stay compatible.
- A safer approach: keep storing tensors but allocate them with shape
  `(B, Hkv, T, D)` of dtype `torch.uint8` (1 byte instead of 2) and never
  read from them.  Saves ~50%.  Or `(B, Hkv, T, 1)` (since only the seq
  dim affects get_seq_length).
- The shape-`(B, Hkv, T, 1)` approach is the cleanest.  Let's try it.

**Risk / cost**:
- 1 file change, no API churn.
- Must verify generation still works (HF model often calls
  `past_key_value.key_cache[i].shape[-2]` and that's it).
- TinyLlama smoke generation test would catch any breakage immediately.

**Expected return**: ~3 GB peak GPU memory reduction at carekv_stored
batch=1 (the dominant memory item).

**Recommendation**: **paper-ready candidate** — see implementation below.

### Implementation outcome

A 6-line edit in `llama_patch.py` would replace the dummy
`hidden_states.new_zeros((B, Hkv, T, D))` with shape `(B, Hkv, T, 1)`
zeros, exploiting HF's reliance on `.shape[-2]` only.  Compatibility risk
is the only blocker: some HF code reads `.shape[-1]` from `past_key_value`
to check D agreement.  In transformers 4.45 the LlamaAttention call chain
does NOT read shape[-1] from past_key_value — but other models/versions
might.  Marked **experimental** for now; a controlled flag
(`CAREKV_LIGHT_HF_CACHE=1`) is straightforward to add.

---

## Overall ranking of remaining ideas

| # | Idea | Status | Expected return | Implementation cost |
|---|---|---|---|---|
| 3 | Vectorized joint+both prefill | **future work, top priority** | 1.4× wall-clock for all paper runs | medium (~1 day) |
| 4 | Lightweight HF Cache | **paper-ready candidate** (flagged) | ~3 GB peak GPU memory | small |
| 1 | Soft K-guided V routing | future work, low priority | unclear; k_first already underperforms | medium |
| 2 | Entropy-aware read budget | future work, low priority | unclear at current budgets | small |

## Conclusion

**For the current paper draft**, the existing optimized config (joint +
score_normalize + cached + packed_base + scale_quant=int8 + abs SK2 SV4
RK2 RV2 + uniform layer budget) is **paper-ready as-is**.  PPL = 3.8294
on synthetic and 13.54 on WikiText-2 INT3, ~0.25× FP16 memory.

**For a follow-up paper version**, prioritize vectorizing joint+both
prefill (#3) — that's the one optimization that materially changes what
evaluations are feasible.  Lightweight HF Cache (#4) is a small,
near-zero-risk memory win once the flag is added.  Coupled routing (#1)
and entropy-aware budget (#2) are speculative and likely won't beat the
current best.

---

## Candidate-cap saturation of the store budget (interpretation, diagnostic)

The absolute store-budget sweep is **saturated by residual granularity**, not
optimized to a global optimum. K/V residual candidates are block/group-level:

```
K candidates per page = head_dim      / k_channel_group = 64 / 32 = 2
V candidates per page = ceil(page_size / v_token_block) = 16 / 4 = 4
```

So `STORE_ABS_K > 2` and `STORE_ABS_V > 4` select from pools that only hold
2 and 4 candidates — the *effective* budget saturates at the cap and PPL cannot
improve (verified: `SK∈{2,4,8}, SV∈{4,8}` all give the same effective budget,
reads, and PPL). `utils.estimate_memory_bytes` clamps stored slots to the same
caps (`utils.py:56-57`).

Implication for claims: `SK=2, SV=4` is the **minimum-storage equivalent under
the current residual granularity**, *not* a proven globally optimal store
budget. To raise the PPL ceiling you must change the granularity (more, smaller
candidates) at a memory cost — see `summaries/budget_granularity_sweep.md` and
`figures/fig_budget_granularity_sweep.png` for the diagnostic granularity
sensitivity sweep. The paper-best config is unchanged pending a granularity
that clearly improves PPL without inflating memory.

Optimization-plan note: a finer-granularity residual (e.g. `k_channel_group=16`
→ K_cap=4, or `v_token_block=2` → V_cap=8) is the only lever that can lift the
store-side PPL ceiling; it is gated on the granularity sweep showing a
favorable PPL-vs-memory trade. Effective-budget columns
(`req_SK/eff_SK/...`/`residual_mem_bytes`) are now emitted on every budget row
so future sweeps surface saturation automatically.
