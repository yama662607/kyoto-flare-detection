# Kyoto Flare Detection Project

[日本語のREADMEはこちら](./README_Ja.md)

## Overview

This project provides a Python framework to detect stellar flares in TESS light curves and analyze their energies and occurrence rates.

### Key Features

- Load light curves from TESS FITS files
- Gap correction and detrending
- Automatic flare detection
- Estimate flare energy, duration, and peak times
- Visualization with Plotly and Matplotlib

## Data Setup

TESS data are large and excluded from Git. Place FITS files under `data/`.
Also place the TESS response file at `data/tess-response-function-v1.0.csv`.

Example structure (filenames vary by sector and target):

```
data/
├── TESS/
│   ├── DS_Tuc_A/
│   │   └── tess2018206045859-s0001-0000000410214986-0120-s_lc.fits
│   ├── EK_Dra/
│   │   └── tess2019198215352-s0014-0000000159613900-0150-s_lc.fits
│   └── V889_Her/
│       └── tess2022164095748-s0053-0000000471000657-0226-s_lc.fits
└── tess-response-function-v1.0.csv
```

## Project Structure

```
/
├── data/                       # Data files (not tracked)
│   ├── TESS/                   # FITS files per star
│   └── tess-response-function-v1.0.csv
├── notebooks/                  # Analysis notebooks
│   ├── flare_create_graphs.ipynb
│   ├── flare_detect_*.ipynb
│   └── learning/
├── outputs/                    # Generated figures and results
│   └── figures/
├── docs/                       # Documentation
├── tools/                      # Utility scripts
├── src/                        # Source code
│   ├── base_flare_detector.py
│   ├── flarepy_*.py
│   └── visualization/
├── .gitignore
├── pyproject.toml
├── justfile
└── uv.lock
```

## Setup

This project uses `uv`.

```bash
uv sync
```

## Usage

Use notebooks in `notebooks/` for analysis. Example:

```python
from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A

file_path = "data/TESS/DS_Tuc_A/tess2018206045859-s0001-0000000410214986-0120-s_lc.fits"

detector = FlareDetector_DS_Tuc_A(file=file_path, process_data=True)

detector.plot_flare()
```

## Outputs

See `docs/OUTPUTS.md` for generated artifacts and debug output locations.
