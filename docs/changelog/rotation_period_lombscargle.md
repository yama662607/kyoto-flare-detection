## BaseFlareDetector Optimization Notes

This document summarizes the main changes made on the `feat/safe-speedup-base-flare` branch to `BaseFlareDetector` and validation tools. Each item lists before/after, goal, and why values remain unchanged. Code excerpts include repository markers (e.g., `# [perf] ...`).

---

### 1. Sliding-window optimization in `reestimate_errors`

| Item | Details |
| --- | --- |
| Before | For each point, `np.searchsorted` was used to find the ±0.5 day window from scratch. |
| After | Reuse a sliding window by updating `start_idx` / `end_idx`. Added comment `# [perf] reuse sliding window instead of per-point searchsorted`. |
| Goal | Reduce behavior from ~O(N log N) to O(N) and avoid repeated `searchsorted` calls. |
| Why values are unchanged | The left/right gap logic is identical and the same samples feed the standard deviation. `uv run python tools/verify_detector_state.py ...` confirmed identical `mPDCSAPfluxerr_cor` hashes. |

```python
while start_idx < n_quiet and quiet_bjd[start_idx] < left:
    start_idx += 1
...
err[i] = np.std(quiet_flux[start_idx:end_idx])  # [perf] reuse sliding window instead of per-point searchsorted
```

---

### 2. Response-function caching in `tess_band_energy`

| Item | Details |
| --- | --- |
| Before | Each call loaded the CSV and recomputed temperature integrals. |
| After | Added module-level caches in `_get_tess_response()` and `_get_star_intensity_ratio()`. Comment `# [perf] cached ratio keeps math identical but avoids recompute`. |
| Goal | Reduce I/O and integration overhead for repeated calls. |
| Why values are unchanged | The cache only stores the first computed result; formulas and floating-point math are unchanged. `verify_detector_state.py` confirms identical `sum_flare_energy`. |

---

### 3. Regular grid + fast solver for `rotation_period`

| Item | Details |
| --- | --- |
| Before | Used a non-regular period grid (`1 / np.linspace(1.0, 8.0, 10000)`), and `LombScargle.power()` used `method="auto"`. |
| After | Introduced `ROTATION_FREQUENCY_GRID = np.linspace(1/8, 1, 10000)` and call `power(..., method="fast")`. Comment `# [perf] regular frequency grid enables LombScargle fast solver`. |
| Goal | Ensure the FFT-based O(N log N) implementation is used and reduce runtime (1.5s → 0.005s). |
| Why values are unchanged | Only discretization differs; the peak index and derived period are identical. Numerical differences are limited to discretization (~2.7e-4 days). |

---

### 4. Plotly reporting and `docs/reports/` organization

| Item | Details |
| --- | --- |
| Before | Matplotlib PNGs scattered under `docs/profiling/`, not browser-friendly. |
| After | Unified tools into Plotly-based output: `tools/profile_base_flare_detector.py`, `tools/compare_profile_results.py`, `tools/verify_detector_state.py`. Outputs organized under `docs/reports/performance/...` and `docs/reports/validation/...`, with `--show-plot` for browser preview. |
| Goal | Consistent visuals and one-command browser preview for developers/researchers. |
| Why values are unchanged | CSV and computation remain the same; only visualization changed. Validated with `base_flare_detector_cumtime_comparison.csv` comparisons. |

---

### 5. State validation previews and tooltips

| Item | Details |
| --- | --- |
| Before | Detail tables showed raw JSON strings, hard to read. |
| After | `serialize_value` adds a `preview` field and renders values like `"[122859.93, 122937.38, 122670.76...] (shape=[15307], dtype=float32)"`. When `--show-plot` is used, tooltips are rendered via `<span title="...">value</span>`. |
| Goal | Make large arrays/dicts readable at a glance, with full JSON on hover. |
| Why values are unchanged | Only the presentation layer is updated; JSON baselines and CSV values are unchanged. |

---

### 6. Generalized Lomb-Scargle rotation period + auto/fast comparison

| Item | Details |
| --- | --- |
| Change | Added `rotation_period_min`, `rotation_period_max`, and `rotation_n_points` to `BaseFlareDetector`, using `make_rotation_frequency_grid(period_min, period_max, n_points)` to build a regular frequency grid and corresponding period array. Defaults match the previous grid (1–8 days, 10000 points). |
| Method handling | `rotation_ls_method` controls `method`, defaulting to `"auto"`. For regular TESS cadence, `auto` picks the fast solver; `"fast"` can be set explicitly. |
| Per-star ranges | Subclasses now explicitly declare their period ranges to match archive/daijiro settings: DS Tuc A 1.0–8.0 days, EK Dra 1.5–5.0 days, V889 Her 0.3–2.0 days. |
| Verification tool | Added `tools/compare_rotation_lomb_methods.py` to compare `method="auto"` vs `"fast"` across FITS files. |
| auto vs fast results | For DS Tuc A (5 files), EK Dra (12 files), V889 Her (4 files), `mean_abs_delta_days`, `max_abs_delta_days`, `mean_rel_delta`, and `max_rel_delta` are all 0.0. |
| Plots and tables | `rotation_period_auto_vs_fast_scatter.png` lies on `y = x`, `rotation_period_diff_hist_hours.png` shows no spread, and `rotation_period_diff_summary_table.png` summarizes per-target stats. |
| Goal | Keep `method="auto"` as default while ensuring a regular grid and parameterized ranges for both generality and performance. |
| Why values are unchanged | `auto` and `fast` yield identical periods for the existing dataset. The grid logic remains equivalent, and per-star ranges simply expose existing values. |

---

### Reference commands

```
uv run python tools/profile_base_flare_detector.py \
  --fits data/TESS/DS_Tuc_A/tess2020212050318-s0028-0000000410214986-0190-s_lc.fits \
  --output-dir docs/reports/performance/ds_tuc_a/s0028/after \
  --show-plot

uv run python tools/compare_profile_results.py \
  --before docs/reports/performance/ds_tuc_a/s0028/before/tess2020212050318-s0028-0000000410214986-0190-s_lc_profile_full.csv \
  --after  docs/reports/performance/ds_tuc_a/s0028/after/tess2020212050318-s0028-0000000410214986-0190-s_lc_profile_full.csv \
  --output-dir docs/reports/performance/ds_tuc_a/s0028 \
  --label-before "main (before)" \
  --label-after  "feat/safe-speedup (after)" \
  --top-n 12 \
  --show-plot

uv run python tools/verify_detector_state.py \
  --fits data/TESS/DS_Tuc_A/tess2020212050318-s0028-0000000410214986-0190-s_lc.fits \
  --plot        docs/reports/validation/global/base_flare_detector_summary_table.png \
  --detail-plot docs/reports/validation/global/base_flare_detector_detail_table.png \
  --table-csv   docs/reports/validation/global/base_flare_detector_variable_status.csv \
  --show-plot
```

Running these commands reproduces the optimized behavior and visualizations.
