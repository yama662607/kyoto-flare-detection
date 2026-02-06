# Flare Detection Pipeline: Consistency Check and Fixes Report

This report summarizes the detailed consistency review between the draft paper (`flare_detection.tex`) and the Python implementation (`flarepy`), along with the fixes applied to ensure scientific validity.

## 1. Identified Inconsistencies and Fixes

### 1.1 Consecutive-point criterion for flare detection
- **Finding**: The legacy code required **3 consecutive** points above $5\sigma$ to identify a flare candidate, while the paper draft specified **2 consecutive** points.
- **Fix**: Updated `BaseFlareDetector.flaredetect()` to use **2 consecutive** points in line with the paper.
- **Scientific rationale**: For 2-minute cadence data, 2 consecutive points are sufficient to reject single-point noise while capturing small flare rises.

### 1.2 Effective observation time calculation
- **Finding**: The legacy code subtracted only large gaps (>= 0.2 day) from the total duration. This incorrectly counted short gaps (e.g., quality-flag removals) as observable time.
- **Fix**: Switched to a cadence-based calculation: `number of points × 2 minutes`.
- **Impact check**:
- **EK Dra (Sector 23)**: reported observation time decreased by about **5.4%**.
- **DS Tuc A (Sector 1)**: decreased by about **5.1%**.
- **Scientific rationale**: The observation time ($T_{\rm obs}$) used for flare frequency ($N/T_{\rm obs}$) should represent intervals where flares are detectable. Excluding short gaps improves frequency estimates.

### 1.3 Energy estimation notation and constants
- **Finding**: The paper draft mixed a time-varying temperature $T_{\rm BB}(t)$ with a fixed $10000\,\mathrm{K}$ assumption. The code also used a slightly simplified physical constant.
- **Fixes**:
- **Paper**: Standardized the notation to a fixed $10000\,\mathrm{K}$ and aligned the integral description with the equations.
- **Code**: Updated the Stefan-Boltzmann constant `sigma_SB` to the accurate value ($5.670374e-5$) and clarified the formula mapping.

## 2. Summary of the Updated Method

| Item | Legacy | Current | Paper reference |
| :--- | :--- | :--- | :--- |
| Flare candidate threshold | 3 consecutive ($5\sigma$) | 2 consecutive ($5\sigma$) | Section 2.1 |
| Effective observation time | Gap subtraction (0.2 day threshold) | Cadence-based ($N \times 2$ min) | Section 2.1 |
| Flare temperature | $10000\,\mathrm{K}$ (fixed) | $10000\,\mathrm{K}$ (fixed) | Section 2.2 |
| Energy formula | Simplified area counting | Standardized intensity ratio | Eq. 11-14 |

## 3. Verification Results
- **Logic tests**: Added `tests/test_logic.py` to verify 2-point detection and the observation-time calculation (all tests pass).
- **Impact check**: For DS Tuc A and V889 Her sectors, the refined observation time slightly increases flare frequency (per day).

---
*Prepared for internal review - January 2026*
