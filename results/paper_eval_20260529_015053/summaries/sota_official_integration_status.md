# Official SOTA integration status (Phase P-direct)

**Goal**: status report on whether each SOTA KV-cache method has been
integrated as an `OfficialAdapter` in the same-condition harness
(`baselines/`).

**Scope of this session**: no official repo was cloned, built, or
executed. Each row below records the **exact blocker** for not doing so
in this session, so the trail isn't lost.

A separate document, `sota_official_repo_feasibility.md`, has the
longer-form repo-by-repo writeup (URLs, dependency risk, model coverage,
expected effort). This document is the **status table** that maps each
method to its current adapter class and integration status.

## Status table

| method | adapter class | status | blocker (concrete) |
|---|---|---|---|
| FP16 reference         | `FP16Adapter`           | ✓ runs (reference)                    | n/a |
| BaseQuant ladder       | `BaseQuantAdapter`      | ✓ runs (same-condition reimplementation) | n/a |
| CARE-KV (paper-best)   | `CAREKVAdapter`         | ✓ runs (this codebase's method)       | n/a |
| KIVI                   | `KIVIStyleAdapter`      | ✓ runs (same-condition reimplementation) | official repo not integrated — uses custom CUDA kernels that need building in this conda env (~1–2 days) |
| KVQuant                | `KVQuantStyleAdapter`   | ✗ unsupported in this turn            | needs pre-RoPE K storage path; opposite of CARE-KV's post-RoPE invariant validated by Phase K-c. ~1–2 days to add a K-store-mode switch through cache.py + llama_patch.py + prefill loop. |
| MiKV                   | `MiKVStyleAdapter`      | ✗ unsupported in this turn            | needs per-token bit-width plumbing through pack/unpack pipeline + a saliency pass at prefill. ~1–2 days. |
| ZipCache               | `ZipCacheStyleAdapter`  | ✗ unsupported in this turn            | saliency pass exists (CARE-KV's router already computes it); blocker is the per-token bit-width storage path, same as MiKV. ~1–2 days. |
| H2O / SnapKV (eviction) | n/a                    | NOT in this comparison                | different axis (token eviction); not comparable on PPL at fixed bit-width. Would belong in a separate "long-context compression" comparison. |
| TurboQuant (emerging)  | n/a                     | NOT in this comparison                | no confirmed runnable official implementation at the time of writing. |

## What "same-condition reimplementation" means here

`KIVIStyleAdapter` is the only non-trivial reimplementation in this turn.
Specifically:

- **Per-channel K (KIVI)** is implemented by monkey-patching
  `transformers.models.llama.modeling_llama.apply_rotary_pos_emb` to
  apply a quant-dequant round trip with `scale = max(|K|_t) / qmax`
  where the max is taken across the time dimension per (B, Hkv, channel).
- **Per-token V (KIVI)** is implemented by wrapping every
  `LlamaAttention.v_proj.forward` to apply a quant-dequant round trip
  with `scale = max(|V|_d) / qmax` where the max is taken across the
  head_dim per (B, Hkv, token).

This is faithful to KIVI's *quantization scheme* but does NOT use
KIVI's custom CUDA kernels for *runtime efficiency*. Therefore:

- **Same-condition reimplementation** is a fair label for the **quality**
  axis (PPL at given bit-width + memory).
- It is **NOT** fair to compare wall-clock latency of CARE-KV
  (PyTorch + Python loops, prototype-latency) against KIVI's actual
  CUDA-kernel speed using this adapter — the adapter has no
  custom kernels either, so it's apples-to-apples on PPL but neither
  is fast.

## How to attempt the deferred integrations later

When a follow-up session has the time budget, recommended order
(easiest first):

1. **KVQuantStyleAdapter** — add a `k_store_mode = "post_rope" | "pre_rope"`
   switch through cache.py and the prefill loop. The current K storage is
   post-RoPE; the new mode would store raw K before RoPE and quantize it,
   then re-apply RoPE on dequant at read time. Largest invasive change.

2. **MiKVStyleAdapter** — add a `bits_per_token` tensor to PageMeta and
   extend the INT2/3/4 packers to honor it. Easier on the cache side,
   harder on the packer side because the existing packers assume uniform
   bit-width per row.

3. **ZipCacheStyleAdapter** — once #2 lands, ZipCache is just a different
   saliency policy on top of the same per-token bit-width infrastructure.
   The saliency computation itself is already free (CARE-KV's residual
   router computes it).

4. **OfficialKIVIAdapter** — clone the official KIVI repo, build the CUDA
   kernels in the conda env, write a thin wrapper that swaps
   LlamaAttention with KIVI's. Once this lands we'd have an
   apples-to-apples *runtime* comparison too, not just PPL.

## Notes for the comparison table reader

When reading `sota_direct_wikitext2_n*.md`:

- Rows tagged **`same-condition reimplementation`** are fair on the **PPL
  axis** at given bit-width and memory budget, and are NOT fair on the
  wall-clock latency axis vs official kernel-based implementations.
- Rows tagged **`unsupported`** show as zero PPL and a blocker reason in
  the `notes` column; they are listed for completeness and to make the
  gap explicit.
- Rows tagged **`reference`** (FP16) are the upper bound on quality and
  the lower bound on memory savings.
- The comparison is **TinyLlama-1.1B only** in this session. Multi-model
  scaling is a separate axis (see `current_bottlenecks_and_optimization_plan.md`
  if present, on `feat/routing-baseline-ablation`).
