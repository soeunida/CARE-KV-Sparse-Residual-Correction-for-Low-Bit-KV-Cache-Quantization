# Multi-model CARE-KV validation (WikiText-2, SL512, N=4)

CARE-KV = paper-best vectorized (SK2 SV4 RK2 RV2, sketch32). **recovery** = (base − carekv)/(base − fp16), fraction of the INT3 quality gap that CARE-KV closes toward fp16. Cells marked ✗ are **infrastructure failures** (OOM / device_map CPU-offload / shared-GPU contention), not method results — they need a clean re-run.

| Model | arch | fp16 | base INT3 | CARE-KV | recovery | router K/V | status |
|---|---|---:|---:|---:|---:|---:|:--|
| NousResearch/Llama-2-13b-hf | MHA (Hkv=40) | 5.6703 | 6.2011 | 6.0931 | 20% | 23850k/28577k | ✓ valid |
| mistralai/Mistral-7B-v0.3 | GQA (Hkv=8) | 6.2493 | 6.8137 | 6.6148 | 35% | 14692k/18861k | ✓ valid |
| upstage/SOLAR-10.7B-v1.0 | GQA (Hkv=8) | 5.7286 | 6.0706 | 5.9413 | 38% | 22452k/27879k | ✓ valid |

## Valid CARE-KV results

- **NousResearch/Llama-2-13b-hf**: base-INT3 6.201 → CARE-KV 6.093 (fp16 5.670) — recovers **20%** of the gap.
- **mistralai/Mistral-7B-v0.3**: base-INT3 6.814 → CARE-KV 6.615 (fp16 6.249) — recovers **35%** of the gap.
- **upstage/SOLAR-10.7B-v1.0**: base-INT3 6.071 → CARE-KV 5.941 (fp16 5.729) — recovers **38%** of the gap.

## Notes
- ✗ cells: shared-server GPU contention + device_map=auto CPU-offload (meta-device) corrupted fp16/base arms on the 34B models (nan/0.0/OOM). Method (CARE-KV) is unaffected where baselines are valid.
- MHA models (Llama-2-13B, Hkv=40) need ≥2 GPUs: the KV-head-indexed CARE-KV cache is ~5× a GQA model's.
- Clean re-run recipe: 3 GPUs per 34B (no offload) + per-arm process isolation (fresh memory between fp16/base/carekv).
