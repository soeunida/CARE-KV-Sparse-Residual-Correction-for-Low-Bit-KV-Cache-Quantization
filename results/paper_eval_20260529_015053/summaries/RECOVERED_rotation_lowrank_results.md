# RECOVERED — rotation + read-breadth + low-rank experiments (consolidated)

> Reconstructed after the previous working tree was deleted. All numbers below
> are from the run logs (/tmp) and are reproducible with the recovered tools.
> TinyLlama-1.1B, WikiText-2 PPL, INT3, paper-best CARE-KV unless noted.
> **Paper-best config UNCHANGED by all of this.**

## 1. TurboQuant vs CARE-KV (fair INT3, audited) — reference
Source: `paper_main_fair_int3_table.csv` (4 models × 3 seq-lens).
- CARE-KV vs **BaseQuant INT3**: 8 win / 3 tie / 0 loss (never worse).
- CARE-KV vs **TurboQuant INT3**: 3 win / 2 tie / 6 loss (competitive, not superior).
- Clearest CARE win: **Mistral-7B SL512 7.12 vs 7.49 (−0.36)**. TurboQuant wins
  DeepSeek-7B and long-context Yi/OpenLLaMA (diffuse-error regime).

## 2. Rotation + CARE-KV stack — screening
Prior pilot: Hadamard **post-RoPE** + CARE-KV = 15.23 ≫ uniform+CARE-KV 13.46 (fail).

N=16 (paper-scale, 2032 tok, noise ≈0.25), RV=2:
| arm | PPL | Δ vs bar |
|---|---:|---:|
| fp16 | 15.757 | −2.13 |
| base_int3 | 22.584 | +6.70 |
| uniform+CARE-KV (bar) | 17.885 | 0 |
| Hadamard pre-RoPE + CARE-KV | 17.915 | +0.03 (tie) |
| random pre-RoPE + CARE-KV | 17.998 | +0.11 |
| Hadamard pre-RoPE standalone | 23.268 | (worse than base) |
| random pre-RoPE standalone | 22.812 | (worse than base) |

**Verdict: NO-GO at RV=2.** H1 confirmed (pre-RoPE ≫ post-RoPE), but the best
stack arm only ties the bar; standalone rotated bases are worse than uniform.

## 3. Coverage ablation (finer granularity vs read-breadth)
N=4 SL=128, full-cap store+read per granularity:
| cell | base | gran | PPL |
|---|---|---|---:|
| uni_g32v4 | uniform | g32v4 (RV4) | 13.195 |
| uni_g16v2 | uniform | g16v2 | 13.342 |
| rot_g32v4 | Hadamard pre | g32v4 (RV4) | **12.917** |
| rot_g16v2 | Hadamard pre | g16v2 | 13.381 |

- **Finer granularity (g32v4→g16v2) REJECTED**: hurts both, worse on rotated
  (diff-of-diff +0.32). Full-cap finer granularity = same residual memory
  (head_dim·page = const), just worse partitioning.
- Best cell rot_g32v4 (read-all V) = 12.917. Coverage lever is **read-breadth**,
  not subdivision. (N=4 — noisy.)

## 4. Read-breadth confirmation — N=16 (the live signal)
g32v4 fixed; vary read budget only (RV2 paper vs RV4 read-all). Equal memory.
| config | base | RK/RV | PPL | Δ vs bar |
|---|---|---|---:|---:|
| uniform RV2 (bar) | uniform | 2/2 | 17.885 | 0 |
| uniform RV4 (read-all) | uniform | 2/4 | 18.106 | **+0.22 (worse)** |
| rotated RV4 (read-all) | Hadamard pre | 2/4 | **17.559** | **−0.33 ✅** |

- Read-breadth **HURTS uniform** (concentrated error; matches "reads>2 add
  noise") but **HELPS rotated** (diffuse error).
- **rotated + read-all = 17.559 beats the bar by −0.33 (above noise)** at equal
  memory; beats uniform+read-all by −0.55. **First config to beat the bar at
  paper scale → WEAK GO.** Mechanism = read-breadth on a diffuse (rotated) base.

## 5. Low-rank dense correction (direction ③) — Phase 0
Eval-mode UPPER BOUND (exact residual-SVD subspace + fp coeffs), base+low-rank
only (no sparse). N=4 SL=128:
| rank | PPL | Δ vs base(16.20) | Δ vs sparse bar(13.46) |
|---|---:|---:|---:|
| 0 (=base) | 16.197 | 0 | +2.74 |
| 1 | 15.764 | −0.43 | +2.30 |
| 2 | 15.939 | −0.26 | +2.48 |
| 4 | 15.141 | −1.06 | +1.68 |
| 8 | 14.023 | −2.17 | +0.56 |

**Verdict: NO-GO.** Even the rank-8 upper bound (14.02) does NOT reach the
sparse CARE-KV bar (13.46), at much higher memory (~+67%/kind). The un-rotated
INT3 residual is **not low-rank** — it is concentrated, which is exactly why
sparse CARE-KV is the bar. Low-rank only fits a *rotated* (diffuse) residual,
where read-breadth already works better.

## 6. 7B confirmation (DeepSeek-7B SL256) — INCOMPLETE
Only base_int3 = 11.19 completed before the working tree was deleted; the CARE
cells (uni_bar, rot_g32v4 read-all) did not finish. **To redo.**

## 7. Coherent picture & next step
- **Un-rotated residual = concentrated → sparse CARE-KV wins** (low-rank loses,
  read-breadth hurts).
- **Rotated residual = diffuse → read-breadth (dense reads) wins** (low-rank
  still loses; finer granularity loses).
- The only above-noise improvement is **rotation (Hadamard pre-RoPE) + read-all
  V** (−0.33 at N=16, TinyLlama). **Next: confirm on 7B diffuse settings
  (DeepSeek-7B, long-ctx Yi/OpenLLaMA)** where TurboQuant wins — beware 7B
  non-transfer (as rotation showed). QJL (TurboQuant's real edge) is score-level
  and cannot be stacked on CARE-KV's value-level correction.
