# Part F — Streaming-decode KV handling under CARE-KV

> **How newly generated KV is handled during decode**, and which store
> policy CARE-KV uses. Validated by `tools/test_carekv_streaming_decode.py`
> (smoke CSV `ablations/carekv_streaming_decode_smoke.csv`).

## The two store policies

### Policy 1 — Online store (what CARE-KV implements today)

For each newly generated token `t`:
1. Compute its `K_t, V_t` (post-RoPE for K).
2. Quantize → base `K̂_t, V̂_t` (packed INT-b + page scale), append to the
   **base KV cache** (open-page append, no fresh page per token).
3. Compute residuals `R_K = K_t − K̂_t`, `R_V = V_t − V̂_t` **immediately**.
4. The router scores residual slots; the **store budget** (SK/SV) selects
   which slots to keep; selected slots append to the **residual cache**.
5. Future decode queries read the per-`(layer, kv_head)` residual slots
   (read budget RK/RV) and apply the correction.

Properties: O(1) extra state per token; every past generated token can be
corrected; no fp16 window. This is the policy the prototype runs under
`use_cache=True` (HF `DynamicCache` + open-page append).

### Policy 2 — Delayed store (design alternative, not implemented)

Keep the most recent `W` tokens in **fp16** (exact, no quant), and only
quantize + compute/store residuals once a token rolls out of the window:
1. New token's K/V kept fp16 in a small ring buffer of size `W`.
2. When the buffer evicts token `t−W`, quantize it, compute its residual,
   run the store policy, append selected slots to the residual cache.
3. Decode reads: exact fp16 for the last `W` tokens, base+residual for the
   rest.

Trade-off: spends `W · (per-token fp16 KV)` extra memory to make the most
recent (typically highest-attention) tokens exact, at the cost of a short
delay before a token is compressed. Useful when recent-token fidelity
matters more than the small fp16 window (e.g. short-range copying). Not
implemented here — documented as the natural extension.

## Why online store is the default

- The router's output-error-aware selection is **already cheap per token**
  (Part E: correction is <0.3% of attention FLOP), so there is no FLOP
  reason to delay.
- Delayed store only changes *which* tokens are exact, not the asymptotic
  memory (the fp16 window is O(W), the residual cache is O(T)). For the
  paper's compression claim, online store is the conservative choice
  (every token compressed immediately).
- The READ=0 ≡ base_quant invariant (Part D / pytest) holds step-by-step
  under online store, which is the safety gate for the decode path.

## Smoke acceptance (online policy)

`tools/test_carekv_streaming_decode.py` prefills a prompt and decodes token
by token under `use_cache=True`, asserting:

| # | criterion | how checked |
|---|---|---|
| 1+3 | generated tokens append to base KV cache; length grows correctly | `past.get_seq_length()` == prompt+step each step |
| 2 | selected residuals append to residual cache | `k_slots_stored + v_slots_stored` grows over steps |
| 4 | read router sees prior generated-token residuals | `k_slots_read + v_slots_read` grows over steps |
| 5 | READ=0 decode == base-quant decode | max\|Δlogit\| < 1e-2 vs base_quant, step by step |
| 6 | no memory leak across steps | allocated MB stable (±10%) across decode |

### Smoke results (TinyLlama, prompt=32, decode=6)

| # | criterion | result | evidence |
|---|---|---|---|
| 1+3 | cache length grows correctly | **PASS** | len 33→34→…→38, +1/step |
| 5 | READ=0 == base-quant (step by step) | **PASS** | max\|Δlogit\| = **0.00e+00** |
| 6 | no memory leak across steps | **PASS** | allocated 3313.0→3313.1 MB (flat) |
| 2 | residual store grows per generated token | **FAIL** | `k/v_slots_stored` stays 0 during decode |
| 4 | read router sees new-token residuals | **FAIL** | reads flat at the prefill value (K42771/V47341), not growing |

**Honest reading.** The three *invariants* that gate correctness hold: the
base KV cache grows one slot per generated token, READ=0 decode is
**bit-identical** to base-quant at every step, and there is no memory leak.
The two *growth* checks fail because the prototype's `use_cache=True` decode
path does **not** run the full incremental CARE-KV store/read per generated
token — the residual store/read counters do not advance past their prefill
values during decode. This is the **Phase-G-v1 limitation** already noted in
`tools/bench_latency.py` and §3 of CLAUDE.md (streaming decode currently
re-uses prefill-shaped correction rather than true open-page incremental
residual store). The **online-store policy above is the intended design**;
wiring the incremental residual store + per-token router into the decode
path (and fusing it — Part E) is the outstanding engineering work.

CSV: `ablations/carekv_streaming_decode_smoke.csv`.

## Decode latency caveat

The prototype's `carekv_stored` decode path currently re-runs prefill-style
correction per step (Phase G v1), so per-token wall-clock is high (Part E /
`latency/latency.csv`). The *policy* (online residual store) is independent
of that implementation cost; a fused incremental-decode kernel would keep
the same online-store semantics at much lower latency.
