# Budget experiments — overview and interpretation limits

**Status: diagnostic.** All numbers below come from the small synthetic
forward-pass eval (TinyLlama-1.1B, INT3 base, `route_policy=joint`,
`score_normalize=1`, `correction_impl=cached`). They are *sanity / diagnostic*
artifacts used to choose and **interpret** the paper-best budget — not
paper-scale dataset results. The paper-best config (§2 of `CLAUDE.md`) is
**unchanged** by this analysis.

Tools:
- `tools/eval_budget_experiments.py` — A1 ratio / A2 absolute / B store / C read
  / D K-V balance sweeps. Each row now carries candidate-cap-aware
  effective-budget columns (see below).
- `tools/eval_budget_granularity_sweep.py` — varies the residual *granularity*
  (`k_channel_group`, `v_token_block`) to move the candidate caps themselves.
- `tools/make_budget_figures.py` — figures, including
  `fig_budget_granularity_sweep.png`.

---

## Effective-budget reporting

Every budget row reports both the **requested** and the **effective** store
budget, plus the per-page candidate caps, store utilization, the actual
recovered residual elements, and residual memory:

| column | meaning |
|---|---|
| `req_SK` / `req_SV` | requested store budget (the `STORE_ABS_K/V` knob) |
| `K_cand_cap` / `V_cand_cap` | per-page candidate cap fixed by granularity |
| `eff_SK` / `eff_SV` | `min(req, cap)` — what the budget can actually select |
| `budget_wasted` | `max(0, req_SK−K_cap) + max(0, req_SV−V_cap)` |
| `store_util` | `(eff_SK+eff_SV) / (K_cand_cap+V_cand_cap)` |
| `recovered_K_elems` | `K_reads × page_size × k_channel_group` |
| `recovered_V_elems` | `V_reads × v_token_block × head_dim` |
| `residual_mem_bytes` / `residual_mem_MB` | residual-only memory at `eff` store budget |

The candidate caps come straight from the store-time enumeration in
`residual_store.py`:

```
K candidates per page = head_dim      / k_channel_group   (residual_store.py:198)
V candidates per page = ceil(page_size / v_token_block)   (residual_store.py:230)
```

For the paper config (`head_dim=64`, `k_channel_group=32`, `page_size=16`,
`v_token_block=4`):

```
K_cand_cap = 64 / 32        = 2
V_cand_cap = ceil(16 / 4)   = 4
```

`utils.estimate_memory_bytes` already clamps stored slots to exactly these caps
(`utils.py:56-57`), so the memory model and the router agree: nothing beyond the
cap is ever stored.

---

## Candidate-cap saturation and interpretation limits

**The store-budget sweep is saturated by candidate granularity, not optimized
to a global optimum.**

- K and V residual candidates are **block/group-level, not scalar-level.** One
  K candidate is an entire `k_channel_group`-wide channel block spanning the
  whole page; one V candidate is an entire `v_token_block`-tall token block
  spanning all `head_dim` channels. The number of such candidates per page is
  fixed by the granularity, independent of the budget knob.
- Under the current granularity there are only **2 K candidates and 4 V
  candidates per page.** Therefore `SK > 2` and `SV > 4` **do not add any new
  candidate** — `eff_SK` saturates at 2 and `eff_SV` saturates at 4. The store
  sweep confirms this directly: `SK=2,SV=4`, `SK=4,SV=4`, `SK=8,SV=4`, and
  `SK=4,SV=8` all collapse to the **same** `eff_SK=2, eff_SV=4`, the same router
  reads, and the **same PPL** (see the store-budget table below and §4 of
  `final_report.md`).
- Consequently, **larger store budgets cannot improve PPL in this setup.** Once
  the budget reaches the cap, the residual store already holds every candidate
  that exists at this granularity; raising the knob is a no-op.

**What this validates, and what it does not:**

- ✅ It validates that `SK=2, SV=4` is the **minimum-storage equivalent under
  the current residual granularity** — i.e. the smallest store budget that
  reaches full candidate coverage. Spending less (`SK<2` or `SV<4`) drops
  candidates and costs PPL; spending more is wasted.
- ❌ It does **not** prove `SK=2, SV=4` is a **globally optimal store budget.**
  The cap, and hence the PPL ceiling, is a property of the chosen granularity.
  To learn whether *more* residual capacity helps, you must change the
  granularity (more, smaller candidates), which is what the granularity
  sensitivity sweep does.

> Wording correction (supersedes earlier phrasing): do **not** describe
> `SK=2, SV=4` as a "globally optimal store budget." The correct claim is
> *"minimum-storage equivalent under the current residual granularity."*

**Read budget is not cap-saturated.** Read budgets are capped by the number of
*stored* slots actually present (`residual_router.py:71-72`). With the paper
store budget there are up to 2 stored K and 4 stored V candidates per page, so
`RK=2` is at the K cap but `RV=2` sits **below** the 4 available V slots. The
finding that `RK/RV > 2` degrades PPL (§4 of `final_report.md`) is therefore a
genuine signal-vs-noise trade-off, **not** a candidate-cap artifact.

---

## Store-budget sweep (B) — effective budget table

`RK=RV=2` fixed; synthetic SL=64, INT3. *(populated from
`ablations/store_budget_sweep.csv`.)*

| label | req_SK | eff_SK | K_cap | req_SV | eff_SV | V_cap | wasted | util | PPL | K_reads | V_reads | recK | recV | res_MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| store_SK0_SV0 |  |  |  |  |  |  |  |  | 140.902 | 0 | 0 |  |  |  |
| store_SK1_SV1 |  |  |  |  |  |  |  |  | 127.8923 | 93635 | 86589 |  |  |  |
| store_SK1_SV2 |  |  |  |  |  |  |  |  | 123.2199 | 68949 | 111275 |  |  |  |
| store_SK2_SV2 |  |  |  |  |  |  |  |  | 106.361 | 93451 | 86773 |  |  |  |
| store_SK2_SV4 |  |  |  |  |  |  |  |  | 102.7457 | 81705 | 98519 |  |  |  |
| store_SK4_SV4 |  |  |  |  |  |  |  |  | 102.7457 | 81705 | 98519 |  |  |  |
| store_SK8_SV4 |  |  |  |  |  |  |  |  | 102.7457 | 81705 | 98519 |  |  |  |
| store_SK4_SV8 |  |  |  |  |  |  |  |  | 102.7457 | 81705 | 98519 |  |  |  |

The rows with `req` budgets above the caps share one `eff` budget, one set of
recovered elements, and one PPL — the saturation made explicit.

---

## Granularity sensitivity sweep (diagnostic)

To test whether finer residuals (more candidates, more memory) actually lower
PPL, `tools/eval_budget_granularity_sweep.py` varies the granularity to move
the caps:

```
k_channel_group ∈ {64, 32, 16}  → K_cand_cap ∈ {1, 2, 4}
v_token_block   ∈ {8,  4,  2}   → V_cand_cap ∈ {2, 4, 8}
```

Each cell stores **all** candidates at its granularity (`SK=K_cap`,
`SV=V_cap` — the minimum-storage equivalent for that granularity) and reads at
the paper budget (`RK=min(2,K_cap)`, `RV=min(2,V_cap)`). Small eval:
TinyLlama, SL=128, INT3, `joint` + `score_normalize=1` + `cached`.

See `summaries/budget_granularity_sweep.md` and
`figures/fig_budget_granularity_sweep.png`. **Marked diagnostic** — it
validates the candidate-cap interpretation; it does not by itself establish a
new paper-best granularity (§5: keep the current paper-best unless a setting
clearly improves PPL without inflating memory).
