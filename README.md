[日本語 (Japanese)](README_Ja.md)

# Kyoto Flare Detection Project

## 🚀 Quick Start

Follow these steps to quickly set up and start your analysis.

### 1. Prerequisites
Install `uv` if you haven't already (see [Setup Guide](#-setup-guide) below for detailed instructions).

### 2. Setup the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yama662607/kyoto-flare-detection.git
   ```
2. **Open the project**:
   Open the `kyoto-flare-detection` folder in your preferred IDE (e.g., **VS Code**, **Cursor**, or **Antigravity**).
3. **Install dependencies**:
   Run the following command in the IDE's integrated terminal to setup the environment:
   ```bash
   uv sync
   ```

### 3. Start Analysis
1. **Prepare Data**:
   Place your TESS FITS data in `data/TESS/<star_name>/`.
2. **Run Notebooks**:
   Open and execute any analysis notebook in the `notebooks/` directory.


## Overview

This project provides a Python framework to detect stellar flares in TESS light curves and analyze their energies and occurrence rates.

### Key Features

- Load light curves from TESS FITS files
- Gap correction and detrending
- Automatic flare detection
- Estimate flare energy, duration, and peak times
- Visualization with Plotly and Matplotlib
- Cross-platform support (macOS, Linux, Windows)

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


## Usage

Use notebooks in `notebooks/` for analysis. Example:

```python
import sys
from pathlib import Path

# --- 1. Python Path Configuration ---
# Get the absolute path of the current directory
PROJECT_ROOT = Path().resolve()

# If running from 'notebooks' or 'src' subdirectories, move up to the project root
if PROJECT_ROOT.name in ['notebooks', 'src']:
    PROJECT_ROOT = PROJECT_ROOT.parent

# Add the project root to sys.path so that 'src' module can be imported correctly
sys.path.insert(0, str(PROJECT_ROOT))

# --- 2. Flare Detection Execution ---
# Import the specific detector class for the target star
from src.flarepy_DS_Tuc_A import FlareDetector_DS_Tuc_A

# Specify the target TESS FITS file path (constructed relative to PROJECT_ROOT)
file_path = PROJECT_ROOT / "data/TESS/DS_Tuc_A/tess2018206045859-s0001-0000000410214986-0120-s_lc.fits"

# Initialize the detector and process data (detrending, flare detection, energy calculation)
# Setting process_data=True triggers the analysis pipeline automatically upon initialization
detector = FlareDetector_DS_Tuc_A(file=file_path, process_data=True)

# Visualize the light curve and detected flares using Plotly (interactive plot)
detector.plot_flare()

# Visualize the energy frequency distribution using Matplotlib (static plot for papers)
detector.plot_energy_matplotlib()
```

## Outputs

See `docs/OUTPUTS.md` for generated artifacts and debug output locations.

---

## 🔧 Setup Guide

### 1. Install uv

Used for fast and reliable Python environment management.

-   **macOS / Linux:**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
-   **Windows:**
    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```
-   **Other Package Managers:**
    -   macOS (Homebrew): `brew install uv`
    -   Windows (winget): `winget install astral-sh.uv`

### 2. Install Just (Optional)

Highly recommended for development and automated maintenance tasks (build, check, analysis tasks, etc.).

-   **macOS (Homebrew):** `brew install just`
*   **Windows (winget):** `winget install casey.just`
*   **Linux (Ubuntu/Debian):** `sudo apt install just`

> [!TIP]
> After installation, restart your terminal and verify that `just --version` and `uv --version` work correctly.

> [!NOTE]
> **Cross-platform Support:** This project uses `Pathlib` for all file path operations, ensuring consistent behavior across macOS, Linux, and Windows. When using `just`, we recommend using a shell that supports standard commands (like Git Bash on Windows) for the best experience with the `justfile`'s utility tasks.
