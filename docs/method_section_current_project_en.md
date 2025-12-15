# Method (Based on the current `src` implementation / Kyoto Flare Detection)

## 1. Overview (analysis pipeline)

We detect stellar flares from TESS light curves (FITS light curve; `*_lc.fits`) and estimate the flare energy, occurrence rate, stellar rotation period, and spot-related proxies. The analysis follows the pipeline implemented in `BaseFlareDetector.process_data()` in `src/base_flare_detector.py`.

- **Primary implementation sources**
  - **Common pipeline**: `src/base_flare_detector.py` (`BaseFlareDetector`)
  - **Star-specific overrides**:
    - `src/flarepy_DS_Tuc_A.py` (`FlareDetector_DS_Tuc_A`)
    - `src/flarepy_EK_Dra.py` (`FlareDetector_EK_Dra`)
    - `src/flarepy_V889_Her.py` (`FlareDetector_V889_Her`)

### 1.1 Processing order of `process_data()`

`BaseFlareDetector.process_data(ene_thres_low=None, ene_thres_high=None, skip_remove=False)` executes the following steps in order:

1. `remove()` (may be overridden in subclasses; skipped when `skip_remove=True`)
2. `apply_gap_correction()` (gap correction and buffer extension)
3. `detrend_flux()` (low-frequency removal and baseline estimation)
4. `reestimate_errors()` (re-estimation of the photometric uncertainties)
5. `flaredetect()` (primary detection, coarse event segmentation, coarse integrated excess)
6. `flaredetect_check()` (event re-validation, refined segmentation, e-folding time, refined integration)
7. `calculate_precise_obs_time()` (effective observing time)
8. `flare_energy(energy_threshold_low, energy_threshold_high)` (event count and total energy within a specified energy range)
9. `flux_diff()` (rotational variability amplitude and spot proxy)
10. `rotation_period()` (rotation period via Lomb–Scargle)

## 2. Data and normalization

### 2.1 Input data (TESS FITS light curve)

`BaseFlareDetector.load_TESS_data()` reads the input FITS file (`file`) via `astropy.io.fits.open(..., memmap=True)` and uses the following columns from extension 1 (`hdulist[1].data`):

- **Time**: `time`
- **Flux**: `PDCSAP_FLUX` by default (may switch to `SAP_FLUX`; see below)
- **Flux uncertainty**: `PDCSAP_FLUX_ERR` or `SAP_FLUX_ERR`

Rows with missing flux values are excluded using `~np.isnan(flux)`.

### 2.2 Switching flux columns by sector threshold

The sector number is extracted from the filename `*_lc.fits`. If and only if `sector_threshold` is set (in a subclass) and the extracted sector number is larger than `sector_threshold`, the pipeline switches to:

- `flux_field = "SAP_FLUX"`
- `flux_err_field = "SAP_FLUX_ERR"`

Otherwise it uses `PDCSAP_FLUX` / `PDCSAP_FLUX_ERR`.

### 2.3 Flux normalization

The loaded flux `pdcsap_flux` and its uncertainty `pdcsap_flux_err` are normalized by a star-specific constant `flux_mean`.

- `norm_flux = pdcsap_flux / flux_mean`
- `norm_flux_err = pdcsap_flux_err / flux_mean`

These normalized series are used in the subsequent processing (detrending, flare detection, rotation period estimation, etc.).

## 3. Gap correction and buffer extension

### 3.1 Gap correction (flux continuity)

`BaseFlareDetector.apply_gap_correction()` computes the time differences `diff_bjd = np.diff(bjd)` for `bjd = self.tessBJD` and identifies gaps as indices where:

- `diff_bjd >= gap_threshold`

For each gap index `idx`, the flux series after the gap is shifted to be continuous with the pre-gap level:

- `flux[idx+1:] -= flux[idx+1] - flux[idx]`

The default `gap_threshold` in `BaseFlareDetector` is `0.1`, but it can be overridden per target in subclasses.

### 3.2 Buffer extension (to mitigate edge effects)

To reduce boundary artifacts in detrending, a buffer of `buffer_size` points is added to both ends of the flux series.

- `flux_ext = [flux[0] repeated buffer_size times] + flux + [flux[-1] repeated buffer_size times]`
- `flux_err_ext = [0.0001 repeated buffer_size times] + flux_err + [0.0001 repeated buffer_size times]`

Time stamps are also extrapolated linearly using `dt_min = 2/(24*60)` days (2 minutes) to build `bjd_ext`, and the extended arrays are stored as:

- `self.gtessBJD = bjd_ext`
- `self.gmPDCSAPflux = flux_ext`
- `self.gmPDCSAPfluxerr = flux_err_ext`

## 4. Detrending (baseline estimation and low-frequency removal)

### 4.1 Low-pass filter (FFT-based)

`BaseFlareDetector.lowpass(x, y, fc)` applies an FFT-based low-pass filter by zeroing the Fourier components above the cutoff `fc` (effectively in units of day$^{-1}$ in the current implementation), and returns the real part of the inverse transform.

- Sampling interval: `dt = 2/(24*60)` days (2 minutes)
- Frequency axis: `freq = linspace(0, 1/dt, n)`

### 4.2 Baseline estimation (cubic spline)

`BaseFlareDetector.detrend_flux()` generates a detrended series as follows:

1. Apply low-pass to the extended series:
   - `filtered_flux = lowpass(time_ext, flux_ext, fc=f_cut_lowpass)`
2. Compute residuals:
   - `s1_flux = flux_ext - filtered_flux`
3. Define candidate points for baseline estimation (intended to exclude flare-like deviations):
   - `fac = 3`
   - `ss_flarecan = where((s1_flux <= flux_err_ext*fac) | (time_ext < time_ext[10]) | (time_ext > time_ext[-11]))`
4. Fit a cubic spline baseline:
   - `baseline_spline = interp1d(time_ext[ss_flarecan], lowpass(time_ext[ss_flarecan], flux_ext[ss_flarecan], fc=f_cut_spline), kind="cubic")`
5. Evaluate the baseline on the original (non-extended) interval `valid_slice = slice(buffer_size, buffer_size + len(self.tessBJD))`:
   - `flux_spline = baseline_spline(time_ext[valid_slice])`
   - `s2mPDCSAPflux = flux_ext[valid_slice] - flux_spline`

The detrended series used for flare detection is `s2mPDCSAPflux`.

### 4.3 Special detrending for V889 Her

`FlareDetector_V889_Her.detrend_flux()` suppresses flare contamination in baseline estimation by masking intervals that show steep positive changes (flare-like behavior) and interpolating over them before applying the low-pass + spline baseline procedure.

- Compute `diff_time = diff(time_ext)` and `diff_flux = diff(flux_ext)`.
- Additionally compute lagged differences for lags `n=2..5`.
- Candidate flare starts are selected using thresholds (e.g., `> 0.01`) and requiring `diff_time < 0.005`.
- For each candidate start `j`, an end index `i` is found such that `abs(flux_ext[j] - flux_ext[i]) < 0.008`, and the interval `[j, i]` is masked.
- A cubic spline is fitted to unmasked samples and used to fill the masked intervals.

The method then proceeds with the low-pass + spline baseline estimation using `f_cut_lowpass` / `f_cut_spline` and outputs `s2mPDCSAPflux`.

## 5. Re-estimation of photometric uncertainties

`BaseFlareDetector.reestimate_errors()` treats samples with

- `quiet_mask = (flux <= 0.005)`

as “quiet” and estimates the local scatter within a ±0.5 day window (`window = 0.5`) around each timestamp `bjd[i]`. The resulting uncertainty array is stored as `mPDCSAPfluxerr_cor`.

- If no quiet samples fall into the window: `err[i] = NaN`.
- If exactly one sample falls into the window: `err[i] = 0.0`.
- Otherwise, the variance is computed using prefix sums and `err[i] = sqrt(var)`.

Finally, the errors are scaled as:

- `err *= mean(self.mPDCSAPfluxerr) / self.err_constant_mean`

## 6. Flare detection (primary detection and event definition)

### 6.1 Primary detection (approximately 5σ)

In `BaseFlareDetector.flaredetect()`, using the detrended series `flux_detrend = s2mPDCSAPflux` and the re-estimated uncertainties `err = mPDCSAPfluxerr_cor`, candidate points are defined as:

- `oversigma_idx = where(flux_detrend >= err*5)`

Candidates are grouped by requiring adjacency (index difference of 1). The first index of each adjacent group is recorded as a seed index `ss_detect`.

### 6.2 Event interval (contiguous region above 1σ)

Define:

- `overonesigma_idx = where(flux_detrend >= err)`

For each seed in `ss_detect`, expand forward and backward along contiguous samples on `overonesigma_idx` to define the event interval.

Quality filters are applied:

- Events too close to the boundaries are discarded:
  - `(n + j) <= 30` or `(n + k) >= len(bjd) - 30`.
- Events with a large time gap nearby are discarded:
  - Let `a = diff_bjd[(n+j-10):(n+k+10)]`; discard if `max(a) >= (2/(24*60))*20` (approximately ≥ 40 minutes).

For accepted events:

- `starttime = bjd[n+j]`
- `endtime = bjd[n+k]`
- `peaktime` is the time of the maximum `flux_detrend` within the interval.

### 6.3 Coarse integrated excess (`count`)

For each event, compute the coarse integrated excess flux:

- `count = sum(flux_detrend[n+j : n+k+1])`

This value is converted to flare energy in the next step.

## 7. Event re-validation (`flaredetect_check`)

After the primary detection, `BaseFlareDetector.flaredetect_check()` redefines the flare interval and derived properties using a local linear baseline.

### 7.1 Local linear baseline (`a_i`, `b_i`)

Pre- and post-event reference windows are defined as:

- `t_pre`: around `stime - 0.025` days (±0.0125 days)
- `t_post`: around `etime + 0.05` days (±0.025 days)

Using the median flux values in these windows (`val_pre`, `val_post`), define:

- `a = (val_post - val_pre)/(t_post - t_pre)`
- `b = val_pre - a*t_pre`

and compute the baseline-subtracted series:

- `flux_diff = flux - (a*bjd + b)`

### 7.2 Refined start/end times

Starting from the index corresponding to `ptime` (`n_peak`), expand forward and backward while `flux_diff >= err` to determine `n_start` and `n_end`.

### 7.3 Acceptance criteria (implementation-level quality filters)

An event is accepted if:

- The refined interval contains at least 2 samples.
- At least 2 samples satisfy `flux_diff - 3*err >= 0` within the interval.

Thus, the refined acceptance effectively requires a ≥3σ excess in at least two cadences.

### 7.4 E-folding decay time

Let `peak_flux` be the maximum of `flux_diff` within the refined interval. The e-folding decay time `edecay` is measured as the time from the peak until `flux_diff` drops below `peak_flux*exp(-1)`.

### 7.5 Refined integrated excess (`count_new`)

For accepted events:

- `count_new = sum(flux_diff[n_start : n_end+1])`

is computed and used for the final energy estimation.

## 8. Flare energy estimation (TESS band)

### 8.1 TESS response function

The TESS response function is loaded from `data/tess-response-function-v1.0.csv` (`TESS_RESPONSE_PATH`) as:

- `wave` (wavelength)
- `resp` (response)
- `dw = [diff(wave), 0]`

### 8.2 Planck function

The blackbody intensity is computed using `BaseFlareDetector.planck(wav, T)`.

### 8.3 Baseline implementation (BaseFlareDetector)

`BaseFlareDetector.tess_band_energy(count)` estimates the flare energy using:

- Time step: `dt = 120.0` s
- Stellar radius: `Rstar = 695510e5 * R_sunstar_ratio` (cm)
- Stefan–Boltzmann constant: `sigma = 5.67e-5`

The response-weighted intensities are:

- `I_star = sum(dw * planck(wave*1e-9, T_star) * resp)`
- `I_ref = sum(dw * planck(wave*1e-9, 10000) * resp)`

Define:

- `star_intensity_ratio = I_star / I_ref`
- `area_factor = (pi * Rstar^2) * star_intensity_ratio`

Then:

- `E = sigma * (10000^4) * area_factor * dt * count`

where `count` is the integrated excess from `flaredetect` or `flaredetect_check`.

### 8.4 DS Tuc A override (including a companion)

`FlareDetector_DS_Tuc_A.tess_band_energy(count)` includes the companion contribution by using an area-weighted sum of intensities:

- Primary radius: `R_primary = Rsun_cm * R_sunstar_ratio`
- Companion radius: `R_companion = Rsun_cm * 0.864`
- Companion temperature: `T_companion = 4700`

Compute:

- `I_main = sum(dw * planck(wave*1e-9, T_star) * resp)`
- `I_comp = sum(dw * planck(wave*1e-9, 4700) * resp)`
- `I_ref = sum(dw * planck(wave*1e-9, 10000) * resp)`

Area-weighted intensity:

- `star_intensity = I_main*R_primary^2 + I_comp*R_companion^2`
- `area_factor = pi * (star_intensity/I_ref)`

Then:

- `E = sigma * (10000^4) * area_factor * dt * count`

## 9. Effective observing time

`BaseFlareDetector.calculate_precise_obs_time()` treats gaps as intervals where:

- `diff_bjd >= 0.2` days

and computes the effective observing time as:

- `precise_obs_time = (bjd[-1] - bjd[0]) - sum(gap_durations)`

## 10. Event count and total energy within an energy range

`BaseFlareDetector.flare_energy(energy_threshold_low, energy_threshold_high)` sorts the event energy array `energy` and computes:

- `flare_number`: the number of events satisfying `energy_threshold_low <= E <= energy_threshold_high`
- `sum_flare_energy`: the sum of energies for those events

In `process_data()`, the thresholds are taken from `ene_thres_low` / `ene_thres_high` arguments if provided; otherwise the instance values `self.ene_thres_low` / `self.ene_thres_high` are used (default range: `5e33` to `2e40` erg).

## 11. Rotational variability amplitude and spot proxy

### 11.1 Variability amplitude

`BaseFlareDetector.flux_diff(min_percent=0.02, max_percent=0.98)` defines the rotational variability amplitude as the 2–98 percentile range of `mPDCSAPflux`:

- `amplitude = P98 - P2`

and stores it as `brightness_variation_amplitude`.

### 11.2 Temperature decrement

The implementation defines the temperature decrement as:

- `d_T_star = 3.58e-5*T_star^2 + 0.249*T_star - 808`

### 11.3 Spot proxies

The code defines the spot-related proxies as:

- `starspot = 2*pi*(R_sunstar_ratio*695510e3)^2 * (T^4/(T^4 - (T-dT)^4)) * amplitude`
- `starspot_ratio = (T^4/(T^4 - (T-dT)^4)) * amplitude`

(Unit conventions follow the implementation; note that the radius factor differs in appearance from the energy-estimation part.)

### 11.4 DS Tuc A override (primary + companion area)

`FlareDetector_DS_Tuc_A.flux_diff()` calls `super().flux_diff()` and then recomputes `starspot` using the combined projected areas of the primary and companion.

## 12. Rotation period estimation (Lomb–Scargle)

`BaseFlareDetector.rotation_period()` uses `astropy.timeseries.LombScargle` with:

- Time: `t = tessBJD - tessBJD[0]`
- Flux: `y = mPDCSAPflux`

A frequency grid is generated over `1/period_max` to `1/period_min` with `rotation_n_points` samples (default: 10000). The Lomb–Scargle power is computed with `method = rotation_ls_method` (default: `"auto"`) and `assume_regular_frequency=True`.

The period at the maximum power is stored as `per`. The uncertainty `per_err` is defined as half the width of the interval where `power > max(power)/2`.

## 13. Star-specific parameters (subclasses)

The following values are hard-coded in the current `src/flarepy_*.py` implementations.

| Star / Detector                     | `R_sunstar_ratio` | `T_star` [K] | `flux_mean` | `err_constant_mean` | `rot_period` [day] | Rotation search [`min`, `max`] [day] | `f_cut_lowpass` | `f_cut_spline` | `sector_threshold` | `gap_threshold` | Notes                                                                                                  |
| ----------------------------------- | ----------------: | -----------: | ----------: | ------------------: | -----------------: | ------------------------------------ | --------------: | -------------: | -----------------: | --------------: | ------------------------------------------------------------------------------------------------------ |
| DS Tuc A (`FlareDetector_DS_Tuc_A`) |              0.87 |         5428 | 119633.9953 |        0.0005505874 |          0.3672258 | [1.0, 8.0]                           |               3 |              6 |                 74 |            0.05 | Overrides `remove()` (transit masking), `tess_band_energy()` (companion), `flux_diff()` (area scaling) |
| EK Dra (`FlareDetector_EK_Dra`)     |              0.94 |         5700 | 249320.3537 |        0.0004111605 |          0.2094793 | [1.5, 5.0]                           |               3 |              6 |                 74 |             0.2 | Only the gap threshold differs from default                                                            |
| V889 Her (`FlareDetector_V889_Her`) |              1.00 |         6550 | 300710.6233 |        0.0003969586 |          0.4398277 | [0.3, 2.0]                           |              30 |             40 |                 90 |           0.004 | Implements a custom `detrend_flux()` (mask + interpolation + baseline estimation)                      |

## 14. Implementation and environment (reproducibility)

### 14.1 Language and main dependencies

Selected dependencies from `pyproject.toml`:

- Python: `>=3.13`
- `astropy` (FITS I/O, Lomb–Scargle)
- `numpy` (arrays, FFT, statistics)
- `scipy` (`interp1d`)
- `matplotlib` / `plotly` (visualization)

### 14.2 Execution entry points

Typical usage:

- `BaseFlareDetector(file=..., process_data=True)` or
- `FlareDetector_*(file=..., process_data=True)`

which triggers `process_data()` and runs the full pipeline.

---

## Appendix A. Mapping between this Method and code functions

- **Data / Normalization**: `load_TESS_data()`
- **Gap correction**: `apply_gap_correction()`
- **Detrending**: `detrend_flux()` (V889 Her: `FlareDetector_V889_Her.detrend_flux()`)
- **Error model**: `reestimate_errors()`
- **Flare detection**: `flaredetect()`, `flaredetect_check()`
- **Energy estimation**: `tess_band_energy()` (DS Tuc A overrides)
- **Observation time**: `calculate_precise_obs_time()`
- **Rotation period**: `rotation_period()`
- **Spot proxy**: `flux_diff()` (DS Tuc A overrides)
