# Multi-model CARE-KV validation (WikiText-2, SL512, N=4)

CARE-KV = paper-best vectorized (SK2 SV4 RK2 RV2, sketch32). **recovery** = (base − carekv)/(base − fp16), fraction of the INT3 quality gap that CARE-KV closes toward fp16. Cells marked ✗ are **infrastructure failures** (OOM / device_map CPU-offload / shared-GPU contention), not method results — they need a clean re-run.

| Model | arch | fp16 | base INT3 | CARE-KV | recovery | router K/V | status |
|---|---|---:|---:|---:|---:|---:|:--|
| 01-ai/Yi-34B | GQA (Hkv=8) | 5.5252 | 5.9000 | 5.8706 | 8% | 11805k/15719k | ✓ valid |
| NousResearch/Llama-2-13b-hf | MHA (Hkv=40) | 5.7767 | 6.5049 | 6.2296 | 38% | 5990k/7116k | ✓ valid |
| codellama/CodeLlama-34b-hf | GQA (Hkv=8) | 6.9614 | 7.4891 | 7.3034 | 35% | 11732k/13433k | ✓ valid |
| mistralai/Mistral-7B-v0.3 | GQA (Hkv=8) | 6.7854 | 7.2972 | 7.1240 | 34% | 3662k/4726k | ✓ valid |
| upstage/SOLAR-10.7B-v1.0 | GQA (Hkv=8) | 5.6968 | 6.1639 | 5.9853 | 38% | 5669k/6913k | ✓ valid |

## Valid CARE-KV results

- **01-ai/Yi-34B**: base-INT3 5.900 → CARE-KV 5.871 (fp16 5.525) — recovers **8%** of the gap.
- **NousResearch/Llama-2-13b-hf**: base-INT3 6.505 → CARE-KV 6.230 (fp16 5.777) — recovers **38%** of the gap.
- **codellama/CodeLlama-34b-hf**: base-INT3 7.489 → CARE-KV 7.303 (fp16 6.961) — recovers **35%** of the gap.
- **mistralai/Mistral-7B-v0.3**: base-INT3 7.297 → CARE-KV 7.124 (fp16 6.785) — recovers **34%** of the gap.
- **upstage/SOLAR-10.7B-v1.0**: base-INT3 6.164 → CARE-KV 5.985 (fp16 5.697) — recovers **38%** of the gap.

## Notes
- ✗ cells: shared-server GPU contention + device_map=auto CPU-offload (meta-device) corrupted fp16/base arms on the 34B models (nan/0.0/OOM). Method (CARE-KV) is unaffected where baselines are valid.
- MHA models (Llama-2-13B, Hkv=40) need ≥2 GPUs: the KV-head-indexed CARE-KV cache is ~5× a GQA model's.
- Clean re-run recipe: 3 GPUs per 34B (no offload) + per-arm process isolation (fresh memory between fp16/base/carekv).
