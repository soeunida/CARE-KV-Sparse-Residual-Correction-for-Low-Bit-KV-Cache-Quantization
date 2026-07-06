# Multi-model CARE-KV validation (WikiText-2, SL512, N=4)

CARE-KV = paper-best vectorized (SK2 SV4 RK2 RV2, sketch32). **recovery** = (base − carekv)/(base − fp16), fraction of the INT3 quality gap that CARE-KV closes toward fp16. Cells marked ✗ are **infrastructure failures** (OOM / device_map CPU-offload / shared-GPU contention), not method results — they need a clean re-run.

| Model | arch | fp16 | base INT3 | CARE-KV | recovery | router K/V | status |
|---|---|---:|---:|---:|---:|---:|:--|
| NousResearch/Llama-2-13b-hf | MHA (Hkv=40) | 5.7767 | 131.2134 | ✗ missing | — | 0k/0k | ⚠ needs re-run |
| codellama/CodeLlama-34b-hf | GQA (Hkv=8) | 6.9614 | 68.2367 | 121.3975 | -87% | 11336k/13829k | ✓ valid |
| mistralai/Mistral-7B-v0.3 | GQA (Hkv=8) | 6.7854 | 80.5982 | 92.7634 | -16% | 3626k/4762k | ✓ valid |
| upstage/SOLAR-10.7B-v1.0 | GQA (Hkv=8) | 5.6968 | 50.7303 | 102.7072 | -115% | 5564k/7018k | ✓ valid |

## Valid CARE-KV results

- **codellama/CodeLlama-34b-hf**: base-INT3 68.237 → CARE-KV 121.397 (fp16 6.961) — recovers **-87%** of the gap.
- **mistralai/Mistral-7B-v0.3**: base-INT3 80.598 → CARE-KV 92.763 (fp16 6.785) — recovers **-16%** of the gap.
- **upstage/SOLAR-10.7B-v1.0**: base-INT3 50.730 → CARE-KV 102.707 (fp16 5.697) — recovers **-115%** of the gap.

## Notes
- ✗ cells: shared-server GPU contention + device_map=auto CPU-offload (meta-device) corrupted fp16/base arms on the 34B models (nan/0.0/OOM). Method (CARE-KV) is unaffected where baselines are valid.
- MHA models (Llama-2-13B, Hkv=40) need ≥2 GPUs: the KV-head-indexed CARE-KV cache is ~5× a GQA model's.
- Clean re-run recipe: 3 GPUs per 34B (no offload) + per-arm process isolation (fresh memory between fp16/base/carekv).
