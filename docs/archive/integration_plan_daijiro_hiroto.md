# Detailed Integration Plan for daijiro / hiroto Folders

## 0. Goals and Assumptions

Integrate Python code from the `daijiro` and `hiroto` folders into the inheritance-based design under `src` and the workflow in `notebooks`. Assumptions:

- Center the integration around the current implementation of `src/base_flare_detector.py`.
- Existing subclasses (`FlareDetector_DS_Tuc_A`, `FlareDetector_EK_Dra`, `FlareDetector_V889_Her`) are minimal and will receive detailed logic from daijiro/hiroto.
- This document is a plan only; it does not implement changes.

## 1. Architecture Comparison

| Area | src | daijiro | hiroto |
| --- | --- | --- | --- |
| Class structure | Base + subclasses | Single class | Single class |
| Plotting | Plotly + Matplotlib (lightweight) | Plotly + Matplotlib (detailed styles) | Plotly (6-panel layout) |
| Pipeline | Stepwise methods with inheritance | Fully implemented per file | Fully implemented + `process_data_2` |
| Response file | `data/tess-response-function-v1.0.csv` | `./tess-energy.csv` | `./tess-energy.csv` |
| Parameters | Partial in subclasses | Many fixed values | Constructor-driven |

### 1.1 BaseFlareDetector (src) highlights

- Provides TESS load, gap correction, detrend, error re-estimation, detection, plotting, and stats.
- Supports Matplotlib and Plotly outputs.
- Energy calculation reads the response CSV under `data/`.

### 1.2 daijiro / hiroto highlights

- **daijiro**: per-star tuning (fixed `flux_mean`, `err_constant` arrays, binary correction, etc.).
- **hiroto**: detailed Plotly visuals, `process_data_2`, and rich utility methods (`diff`, `remove`, etc.).

## 2. Gap Analysis (by feature)

| Feature | Current src | daijiro / hiroto | Plan |
| --- | --- | --- | --- |
| FITS normalization | `flux_mean` passed at init | Uses measured mean or fixed values | Allow "mean if unset", still accept overrides |
| SAP/PDCSAP switch | Threshold at `data_number > 74` | V889 uses > 90 | Add hook to override per subclass |
| Gap threshold | Fixed 0.05 days | V889: 0.004, hiroto: 0.1 | Add `gap_threshold` parameter and set in subclasses |
| Detrend | FFT low-pass + spline | V889 removes flare candidates first | Override in V889 subclass, reuse common logic |
| Error re-estimation | `err_constant_mean` scaling | daijiro uses measured list | Allow arrays/series injection from subclasses |
| Flare detection | Same 5σ to 1σ logic | Similar | Keep common implementation |
| Post-processing | `flaredetect_check` | Similar | No extra changes |
| Energy | Single-star area, fixed response path | DS Tuc A uses binary sum | Override in DS Tuc A; make response path configurable |
| Plotting | Plotly 2-panel + Matplotlib 2-panel | Detailed styles | Add optional detailed mode |
| Metrics | Starspot + rotation period | Similar | Keep base implementation |
| `process_data_2` | Not present | hiroto has no-remove branch | Add flag or override in subclass |

## 3. Refactor Plan

### 3.1 BaseFlareDetector extensions

1. **Normalization strategy**
   - If `flux_mean` is unset, compute mean from valid data.
2. **Flux column selection hook**
   - Add `_select_flux_columns(data_number)` for subclass overrides.
3. **Gap threshold parameter**
   - Accept `gap_threshold` in `__init__` and use in `apply_gap_correction`.
4. **Error re-estimation flexibility**
   - Allow subclasses to supply scaling series or arrays.
5. **Response CSV path**
   - Accept `tess_response_path` in `__init__` with a default value.
6. **Detailed plot modes**
   - Add `plot_flare(mode="detailed")` for the 6-panel Plotly layout.
7. **Process control flags**
   - Add `skip_remove` / `custom_pipeline` to cover `process_data_2` behavior.

### 3.2 Subclass tasks

| Class | Current | Required tasks |
| --- | --- | --- |
| `FlareDetector_DS_Tuc_A` | Parameters only | Replace `tess_band_energy` with binary version, align `remove` logic, pass `tess_response_path` if needed |
| `FlareDetector_EK_Dra` | Parameters only | Preserve measured error array and validate `flux_mean` strategy |
| `FlareDetector_V889_Her` | Partial detrend override | Set gap threshold to 0.004, ensure low-pass parameters match, port full pre-spline logic |

### 3.3 hiroto-specific features

- `remove` / `no_remove` / `diff` / `calculate_precise_obs_time` largely exist in the base. Align differences and expose `process_data_2` behavior via a flag.

### 3.4 Notebook updates

1. Update imports to use new subclasses.
2. Pass extra parameters (`gap_threshold`, `use_matplotlib_plots`, etc.).
3. Update Plotly cells for detailed mode where needed.
4. Replace legacy `FlareDetector` usage.

### 3.5 Test strategy

- Unit tests per subclass:
- Check normalization with a sample FITS file
- Verify gap threshold handling
- Compare detrend output with snapshots
- Compare energy output with daijiro results (mocked data)
- Integration tests: run notebooks via `papermill` and compare key intermediate metrics.

## 4. Recommended Implementation Order

1. Base class extensions
2. Subclass feature parity
3. Absorb hiroto features + API cleanup
4. Notebook updates
5. Automated tests

## 5. Risks / Considerations

- Fixed vs dynamic `flux_mean` affects detection sensitivity
- Response file path consistency is required to avoid FileNotFound errors
- V889-specific parameters must stay isolated in its subclass
- `process_data` API extensions should be keyword-only to keep compatibility

## 6. Summary

This plan reuses daijiro/hiroto logic while strengthening the base class for a consistent, extensible flare-detection framework. Subclasses should contain only star-specific differences, and notebooks/tests should verify behavior at each step.
