# Comparison Between the hiroto Folder and src

## Overview

This document compares the contents of the `hiroto` folder and the `src` folder to clarify their differences.

## Folder Structure

### hiroto folder
```
hiroto/
├── flare_DS_Tuc_A.ipynb (9.7MB)
└── flarepy_improved.py (41KB)
```

### src folder
```
src/
├── __init__.py
├── __pycache__/
├── base_flare_detector.py (22KB)
├── flarepy_DS_Tuc_A.py.py (1.6KB)
├── flarepy_EK_Dra.py (785B)
├── flarepy_V889_Her.py (4.4KB)
└── archive/
    ├── flarepy_DS_Tuc_A.py (44KB)
    ├── flarepy_EK_Dra.py (43KB)
    ├── flarepy_EK_Dra_plotly.py (44KB)
    └── flarepy_V889_Her.py (52KB)
```

## Key Differences

### 1. Architecture

#### hiroto folder (single-class design)
- **flarepy_improved.py**: contains a single `FlareDetector` class
- Generic design with stellar parameters passed via the constructor
- Intended to be run from a Jupyter notebook

#### src folder (object-oriented design)
- **base_flare_detector.py**: defines the base class `BaseFlareDetector`
- **Star-specific classes**: dedicated subclasses per target
- `FlareDetector_DS_Tuc_A`
- `FlareDetector_EK_Dra`
- `FlareDetector_V889_Her`
- Emphasizes modularity and reuse

### 2. Code scale

| File | hiroto folder | src folder |
|---------|----------------|-------------|
| Main class | 41KB (1,086 lines) | 22KB (490 lines) + star-specific subclasses |
| Jupyter notebook | 9.7MB (321,242 lines) | None |
| Total | ~9.8MB | ~166KB |

### 3. Functional differences

#### hiroto folder
- **Comprehensive plotting**: 6-panel subplot visualization
- **Experimental code**: flags such as `process_data_2()`
- **Self-contained**: all functionality in a single file

#### src folder
- **Modular design**: base class + per-star subclasses
- **Matplotlib support**: in addition to Plotly
- **Improved error handling**: more robust error checks
- **Star-specific tuning**: parameters customized per target

### 4. Implementation differences

#### Data loading
```python
# hiroto folder
match = re.match(r"(.+)-s_lc\.fits$", fname_base)

# src folder
match = re.match(r"(.+)-\d+-\d+-s_lc\.fits$", fname_base)
match = re.match(r"[a-z]+\d+-s00(.+)-\d+-\d+-s_lc\.fits$", fname_base)
```

#### TESS response file path
```python
# hiroto folder
wave, resp = np.loadtxt(".\\tess-energy.csv", delimiter=",").T

# src folder
wave, resp = np.loadtxt("data/tess-response-function-v1.0.csv", delimiter=",").T
```

#### Gap-detection threshold
```python
# hiroto folder
gap_indices = np.where(diff_bjd >= 0.1)[0]

# src folder
gap_indices = np.where(diff_bjd >= 0.05)[0]
```

### 5. Class variable differences

#### hiroto folder
```python
array_flare_ratio = np.array([])
array_energy_ratio = np.array([])
array_amplitude = np.array([])
average_flare_ratio = 0.0
array_observation_time = np.array([])
```

#### src folder
```python
array_flare_ratio = np.array([])
array_observation_time = np.array([])
array_energy_ratio = np.array([])
array_amplitude = np.array([])
array_starspot = np.array([])
array_starspot_ratio = np.array([])
array_data_name = np.array([])
array_per = np.array([])
array_per_err = np.array([])
```

### 6. Features added in src only

- **Starspot metrics**: `starspot`, `starspot_ratio`
- **Rotation period**: `rotation_period()`
- **Lomb-Scargle analysis**: uses `astropy.timeseries.LombScargle`
- **Matplotlib plots**: `plot_flare_matplotlib()`, `plot_energy_matplotlib()`
- **Energy threshold handling**: improved `flare_energy()` logic

### 7. V889 Her-specific implementation

The src folder includes a custom detrending method for V889 Her:
- `difference_at_lag()`
- Multi-lag difference detection
- Flare candidate start/end detection logic

## Development maturity

### hiroto folder
- **Prototype stage**: experimentation inside a Jupyter notebook
- **Single-file design**: prototype-centric
- **Windows paths**: hard-coded Windows path (`".\\tess-energy.csv"`)

### src folder
- **Production-ready**: clean, modular architecture
- **Cross-platform paths**: Unix-friendly paths
- **Maintainability**: separation of concerns and reuse

## Summary

| Item | hiroto folder | src folder |
|------|----------------|-------------|
| Purpose | Prototype / experimentation | Production / maintenance |
| Design | Single-class | Inheritance-based |
| Readability | Medium | High |
| Maintainability | Low | High |
| Functionality | Core + experimental | Extended + stable |
| Size | Large (includes notebook) | Small (modular) |

**Conclusion**: The `hiroto` folder is an early, experimental implementation. The `src` folder represents a refined production codebase that follows object-oriented principles and provides improved modularity, maintainability, and extensibility.
