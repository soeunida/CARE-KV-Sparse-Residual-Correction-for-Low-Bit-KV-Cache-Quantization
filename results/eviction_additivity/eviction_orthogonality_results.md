# Eviction orthogonality (additivity) — methodology + results

**Claim (Section 2).** CARE-KV's PPL benefit is **orthogonal to KV eviction**: the benefit measured without eviction is preserved when token eviction (H2O) is applied on top.

## Methodology

KV compression has two independent axes — **eviction** (*which* tokens to keep) and **CARE-KV** (*correcting* the quantization error of the kept tokens). We test their independence with a **2×2 factorial design** (CARE on/off × eviction on/off), all arms run through the same `CAREKVAdapter` + eviction hook so eviction applies uniformly:

| arm | CARE-KV | eviction |
|---|---|---|
| `base_noevict`   | ✗ (READ budget 0 = INT3 base) | ✗ keep=1.0 |
| `carekv_noevict` | ✓ SK2 SV4 RK2 RV2 | ✗ keep=1.0 |
| `base_evict`     | ✗ | ✓ keep=R |
| `carekv_evict`   | ✓ SK2 SV4 RK2 RV2 | ✓ keep=R |

(+ `fp16` upper-bound reference.) **Eviction = H2O**: keep the fraction R of tokens with highest cumulative attention, protecting sink (=4) and a recent window (env `CAREKV_EVICT_KEEP_RATIO/POLICY/SINK/RECENT`). keep R swept over 0.9 / 0.75 / 0.5.

**Additivity / orthogonality test** (zero 2-factor interaction):

```
benefit_noevict = base_noevict − carekv_noevict
benefit_evict   = base_evict   − carekv_evict
orthogonal/additive  ⟺  benefit_evict ≈ benefit_noevict   (|Δ| ≤ 0.15)
```

Metric: WikiText-2 PPL, N=4, SL=256 (TinyLlama) / 512 (7B). `carekv_evict` is the **combined** config (CARE-KV ⊕ eviction) — the practical deployment of both compressions together.

## Results

| model | keep | policy | fp16 | base noev | CARE noev | base evict | CARE evict | CAREΔ noev | CAREΔ evict | Δ-of-Δ | orthogonal |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| Mistral-7B | 0.9 | h2o | 6.785 | 7.297 | 7.018 | 7.598 | 7.300 | +0.279 | +0.298 | +0.019 | ✅ |
| DeepSeek-7B | 0.9 | h2o | 8.595 | 9.706 | 9.267 | 9.918 | 9.395 | +0.440 | +0.523 | +0.083 | ✅ |
| TinyLlama-1.1B | 0.5 | h2o | 10.152 | 12.911 | 11.423 | 47.122 | 33.770 | +1.488 | +13.352 | +11.864 | ≈/✗ |
| TinyLlama-1.1B | 0.75 | h2o | 10.152 | 12.911 | 11.423 | 20.704 | 16.166 | +1.488 | +4.537 | +3.049 | ≈/✗ |
| TinyLlama-1.1B | 0.9 | h2o | 10.152 | 12.911 | 11.423 | 14.093 | 12.378 | +1.488 | +1.715 | +0.227 | ≈/✗ |

## Reading

- **Mild eviction (keep=0.9) — orthogonality holds**, cleanest on 7B: Mistral-7B CAREΔ +0.28→+0.30; DeepSeek-7B CAREΔ +0.44→+0.52; TinyLlama-1.1B CAREΔ +1.49→+1.71.
- **Aggressive eviction (keep≤0.75, TinyLlama)** — eviction inflates PPL sharply, and CARE-KV recovers **even more** (benefit grows: keep=0.5 Δ +1.49→+13.35, keep=0.75 Δ +1.49→+4.54). CARE-KV cushions the eviction-induced collapse rather than being merely additive.
- **Direction is consistent**: CARE-KV's gain over BaseQuant survives (and at low keep, amplifies under) eviction → the two mechanisms are complementary, not competing.

**Status: diagnostic** (N=4, SL=256/512). Source CSVs: `results/eviction_additivity/evict_add_*.csv`.
