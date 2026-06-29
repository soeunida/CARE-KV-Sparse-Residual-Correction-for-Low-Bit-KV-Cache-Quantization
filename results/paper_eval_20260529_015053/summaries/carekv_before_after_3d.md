# CARE-KV vs base_quant — K/V error decomposition (Phase K-e)

Per-layer error diagnostics that make the CARE-KV residual restoration
unambiguously visible. The **2D heatmaps are the primary paper figure**;
3D surfaces and scatters are secondary visual aids.

> Global K/V magnitude plots can look unchanged because CARE-KV only
> restores sparse selected residual slots — the dominant outlier ridges
> survive base_quant intact and overwhelm the visual delta. The
> correction is best visualized through **error maps**, **signed error
> reduction**, and **recovered residual masks**, which is what these
> figures show.

**Tool:** `tools/make_carekv_before_after_3d.py`

## Plot modes

| `--plot-mode`         | 3D output (per layer per kind) | 2D heatmap (per layer per kind) | Role |
|---|---|---|---|
| **`paper-clean`** (NEW) | — | **3-panel** `paper_carekv_{K,V}_error_layer{NN}.png`  + combined **2×3** `paper_carekv_KV_error_layer{LL}.png` | **paper-ready compact** |
| `error-decomposition` (default) | 4-panel `3d_carekv_{K,V}_error_layer{NN}.png`              | **5-panel** `heatmap_carekv_{K,V}_error_layer{NN}.png`  | diagnostic 5-panel heatmap |
| `visible-error`                 | 5-panel `3d_carekv_{K,V}_visible_layer{NN}.png`            | 5-panel `heatmap_carekv_{K,V}_visible_layer{NN}.png`    | supplementary visible |
| `clean-error`                   | 3-panel `3d_carekv_{K,V}_error_clean_layer{NN}.png` (sparse scatter for recovered) | 5-panel primary heatmap (same as above) | supplementary 3D |
| `both`                          | error-decomposition + visible-error |  | bulk |
| `all`                           | all of the above                    |  | bulk |

## Paper-ready compact figure (`paper-clean` mode)

**Per-layer compact (3 panels — `paper_carekv_{K,V}_error_layer*.png`):**

| Panel | Title | Content | cmap / norm | Colorbar |
|---|---|---|---|---|
| 1 | "Base INT3 error"    | `|X_fp − X_hat|`             | `magma`, `vmin=0`, `vmax = percentile(base, 99)` | shared with panel 2 (label "error") |
| 2 | "After CARE-KV"      | `|X_fp − X_care|`            | same as panel 1                                 | shared with panel 1 |
| 3 | "Error reduced"      | `max(|X_fp − X_hat| − |X_fp − X_care|, 0)` | `Reds` + **`PowerNorm(gamma=0.5)`** (boosts small positive reductions); `vmax = percentile(positive>0, 97)` | separate, label "reduction" |

Panel 3 additionally has:
- **Subtle dark-red contour** (`#8B0000`, linewidth 0.35, alpha 0.4)
  outlining only the **top 2 % of recovered cells by magnitude**
  (not every recovered point) — controlled by `--mask-contour-top-percent` (default 2.0)
- Small inset label `"positive = improved"` (upper-left, 8 pt) to disambiguate the sign convention

Typography:
- Title: 16 pt (headline)
- Annotation line: 13 pt
- Subplot title: 13 pt
- Axis labels: 11 pt
- Ticks: 9 pt

Layout polish:
- Title strip rendered on a phantom subplot via `gridspec` so
  `constrained_layout` reserves exactly its vertical share
  (no overlap with subplot titles, distinct fontsizes for title vs annotation)
- Only panel 0 has the `Token` y-label; panels 1 + 2 keep tick numbers but
  no label
- All panels share the `Channel` x-label

CLI knobs added for this mode:
```
--reduction-percentile 97        # panel 3 vmax percentile (97 = better small-value visibility)
--reduction-gamma 0.5            # PowerNorm gamma on panel 3 (smaller = more boost)
--mask-contour-top-percent 2.0   # only top X% of recovered cells get contour overlay
--paper-combined-layer 11        # which layer goes into the combined K/V figure
```

**Combined K/V figure (2 rows × 3 cols — `paper_carekv_KV_error_layer{LL}.png`):**
Same 3-column layout, K on top row + V on bottom row, separate K and V
colorbars (their magnitudes differ ~50×). Default representative layer is
11. All the panel-3 polish (gamma stretch + sparse contour + inset label)
applies per row.

## Diagnostic 5-panel heatmap (`error-decomposition` mode)

**Primary 2D heatmap (5 panels — `heatmap_carekv_{K,V}_error_layer*.png`):**
1. `|X_fp − X_hat|`               — base_quant error (inferno)
2. `|X_fp − X_care|`              — CARE-KV remaining error (inferno, **shared vmin/vmax with 1**)
3. **positive-only** error reduction `max(|X_fp − X_hat| − |X_fp − X_care|, 0)`
   (Reds; `vmax = percentile(positive_reduction, 99)`).
   **Contour overlay** of the recovered-residual binary mask boundary in black.
4. `|X_care − X_hat|`             — recovered residual magnitude (hot)
5. **Recovered residual binary mask** — white background, dark-red ink

Stats annotation under the title:
`"mean error: <base> → <care>   reduction <%>   recovered <%>"`.

**4-panel 3D `error-decomposition` (`3d_carekv_{K,V}_error_layer*.png`):**
1. base error, 2) CARE-KV remaining, 3) signed reduction (diverging RdBu_r), 4) recovered

**3-panel 3D `clean-error` (`3d_carekv_{K,V}_error_clean_layer*.png`):**
1. base error (`plot_surface`)
2. CARE-KV remaining (`plot_surface`, shared scale)
3. recovered as **`scatter3D`** of top 1 % cells — no dense "solid wall" artifact

**5-panel `visible-error`:** prepends `|X_fp|` (with optional red overlay).

## Color design (latest revision)

**Primary 2D heatmap (publication polish):**

| Panel | cmap | range | note |
|---|---|---|---|
| 1 base error | `inferno` | `[0, percentile(base_error, 99)]` | sequential |
| 2 CARE-KV remaining | same cmap | **same vmin/vmax as panel 1** | care error visibly dimmer than base |
| 3 positive reduction | `Reds` (auto-routed when `--reduction-cmap` ∈ sequential set) | `[0, percentile(positive_reduction, 99)]` | + black contour outline of binary mask |
| 4 recovered residual | `hot` | `[0, percentile(recovered, 99)]` | high-contrast sparse spots |
| 5 recovered mask | `{white, dark-red}` `ListedColormap` | `{0, 1}` | binary; alignment check for panels 3+4 |

**3D `error-decomposition` panel 3:**

| Panel | cmap | range | note |
|---|---|---|---|
| 3 signed reduction | `RdBu_r` | `TwoSlopeNorm(-span, 0, +span)` with `span = percentile(|reduction|, 99)` | diverging |

- `--log-error` is **off by default**. When set, plots `log1p(error * --error-gain)`.
- `--error-gain` defaults to 1.0 (no amplification).
- `--overlay-recovered-mask` (visible-error only) scatters the top
  `--overlay-top-percent` (default 1.0%) recovered cells on `|X_fp|`.

## Per-layer summary (from `/tmp/carekv_err_stats_v3.json`)

Setup: TinyLlama-1.1B, Hkv=4, head_dim=64, SL=512, INT3 (`group_size=32`),
paper-best store-side knobs (`page_size=16, STORE_ABS_K=2, STORE_ABS_V=4`,
L2-norm scoring). Stats over the full `(T=512, Hkv·Hd=256) = 131 072`-cell
matrix per layer per kind.

### K (post-RoPE)

| Layer | base mean | base max | care mean | care max | rel. redux | recovered count | sparsity | pos-redux ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.2477 | 1.9717 | **0.1974** | 1.9622 | **20.3 %** | 16 384 / 131 072 | 12.5 % | 12.1 % |
| 11 | 0.7710 | 2.9883 | **0.6494** | 2.9609 | **15.8 %** | 16 384 / 131 072 | 12.5 % | 12.2 % |
| 21 | 0.7804 | 3.2188 | **0.6625** | 3.2188 | **15.1 %** | 16 384 / 131 072 | 12.5 % | 12.2 % |

### V

| Layer | base mean | base max | care mean | care max | rel. redux | recovered count | sparsity | pos-redux ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0  | 0.00232 | 0.0314 | **0.00160** | 0.0138 | **31.0 %** | 32 768 / 131 072 | 25.0 % | 24.3 % |
| 11 | 0.04683 | 0.2155 | **0.03229** | 0.1685 | **31.1 %** | 32 768 / 131 072 | 25.0 % | 24.3 % |
| 21 | 0.10408 | 0.5015 | **0.07025** | 0.4663 | **32.5 %** | 32 768 / 131 072 | 25.0 % | 24.3 % |

Columns:
- `rel. redux = mean(err_base − err_care) / mean(err_base)` — headline metric.
- `recovered count = mask.sum()` — cells whose stored residual is restored.
- `pos-redux ratio = (err_base > err_care).mean()` — fraction strictly improved
  (matches sparsity within rounding because every restored cell strictly
  reduces error and unrestored cells are unchanged).

## Observations

1. **CARE-KV reduces mean error at every layer for both K and V**, with V
   improving roughly **2× more than K** in relative terms (15–20 % vs
   31–33 %). The new 2D heatmap makes this immediately visible: panel 2
   (CARE-KV) looks visibly dimmer than panel 1 (base) at the same scale.

2. **Panel 3 (signed reduction)** is dominated by clear red horizontal
   bands at the stored token rows — CARE-KV improves error at exactly
   those positions, with rare isolated blue cells (small regressions
   where the residual's sign happens to push the wrong way).

3. **Panel 4 (recovered residual)** shows the sparse selection pattern
   directly. K: per-channel-group × per-page top-2 → ~12 % of cells lit.
   V: full-row stripes at the selected tokens → ~25 % of token rows lit.

4. **The clean 3D scatter (`*_error_clean_layer*.png` panel 3)** replaces
   the prior dense bar-3D "solid wall" with a sparse point cloud showing
   only the top 1 % recovered cells. This makes the localization
   immediately readable in 3D as well.

## Caveats

- Decode-time read budget (`READ_ABS_K=2, READ_ABS_V=2`) is not applied
  — these surfaces show the store-side upper bound (what CARE-KV could
  recover if every stored slot were read at decode).
- `K_care` is exact substitution at selected cells, not the first-order
  `ΔO_K` correction used in the actual attention math.
- Single prompt — outlier identity is roughly prompt-invariant in spot
  checks but a multi-prompt average would strengthen the claim.
- Visualization knobs (`--error-gain`, `--log-error`, `--overlay-*`,
  `--*-cmap`, `--*-percentile`) affect only the rendered figures, not the
  underlying numeric stats.
