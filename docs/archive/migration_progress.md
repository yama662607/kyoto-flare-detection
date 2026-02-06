# Flare Detection Code Integration Progress Log

**Created**: 2025-11-13
**Last Updated**: 2025-11-13
**Goal**: Integrate code from `daijiro/` and `hiroto/` into `src/` and archive the originals

---

## Work Overview

### Goals

- Migrate standalone implementations into the `src/base_flare_detector.py`-based architecture.
- Preserve scientifically critical behavior.
- Eventually move `daijiro/` and `hiroto/` into an archive location.

### Target Files

| Star | daijiro impl | hiroto impl | src impl | Status |
| -------- | ------------ | ----------- | -------- | --------- |
| DS Tuc A | ✅ | - | ✅ | 🔄 needs integration |
| EK Dra | ✅ | - | ✅ | ✅ done |
| V889 Her | ✅ | ✅ | ✅ | 🔄 needs verification |

---

## Completed Work

### 1. BaseFlareDetector improvements (completed)

**Date**: 2025-11-13

#### Matplotlib plotting upgrades

- Updated `plot_flare_matplotlib()` to a publication-quality style
- Directly set rcParams
- Embedded fonts via `pdf.fonttype = 42`
- Added configurable `save_path`
- Figure size: `(13, 8)`
- Flare peak markers: `ymin=0.8, ymax=0.85`

- Updated `plot_energy_matplotlib()` similarly
- Removed `_get_matplotlib_style()` (unused)

**Files**:
- `src/base_flare_detector.py`

**Status**: ✅ implemented, tests pending

---

### 2. EK Dra integration analysis (completed)

**Date**: 2025-11-13

**Conclusion**: `src/flarepy_EK_Dra.py` required no changes.

**Reasoning**:
- Correct inheritance from `BaseFlareDetector`
- Parameter settings already match daijiro
- No star-specific behavior required (single star)

**Key parameters verified**:
- `R_sunstar_ratio`: 0.94
- `T_star`: 5700 K
- `flux_mean`: 249320.35370300722
- `err_constant_mean`: 0.0004111604805261475
- `rot_period`: 0.2094793179536128

**Status**: ✅ analysis complete

---

## In Progress

### 3. Sector-based flux column switch (completed)

**Date**: 2025-11-13

**Summary**:
- Added `sector_threshold` to `BaseFlareDetector`.
- Implemented SAP/PDCSAP selection in `load_TESS_data()`.
- Set per-target thresholds:
- DS Tuc A: 74
- EK Dra: 74
- V889 Her: 90

**Additional changes**:
- DS Tuc A overrides `tess_band_energy()` for binary contribution.
- `flux_diff()` uses combined primary+companion area.
- Added `skip_remove` flag to `process_data()`.
- Added `run_process_data_2` in constructor to support no-remove workflow.

**Files**:
- `src/base_flare_detector.py`
- `src/flarepy_DS_Tuc_A.py`
- `src/flarepy_EK_Dra.py`
- `src/flarepy_V889_Her.py`

**Status**: ✅ implemented, tests pending

---

## Planned Work

### 4. DS Tuc A integration (completed)

**Date**: 2025-11-14

**Summary**:
- Overrode `tess_band_energy()` with binary contribution.
- Overrode `flux_diff()` to use combined primary+companion area.
- Implemented `remove()` with 12 transit windows.
- Set DS Tuc A-specific `gap_threshold = 0.05`.

**Status**: ✅ implemented, verification pending

---

### 5. V889 Her integration (completed)

**Date**: 2025-11-14

**Summary**:
- Ported custom detrending logic and gap threshold (0.004).
- Adjusted low-pass parameters (30/40).
- Verified candidate detection logic.

**Status**: ✅ implemented, verification pending
