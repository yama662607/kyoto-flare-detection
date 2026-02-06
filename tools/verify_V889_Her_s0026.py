
import sys
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.flarepy_V889_Her import FlareDetector_V889_Her

fits_file = PROJECT_ROOT / "data/TESS/V889_Her/tess2020160202036-s0026-0000000471000657-0188-s_lc.fits"

detector = FlareDetector_V889_Her(
    file=str(fits_file),
    process_data=True,
    ene_thres_low=5e33
)

print(f"Total flares detected: {len(detector.energy)}")
print(f"Flares >= 5e33: {detector.flare_number}")

# Print detected flare details
if detector.starttime is not None:
    for i in range(len(detector.starttime)):
        status = "OK" if detector.energy[i] >= 5e33 else "LOW"
        print(f"Flare {i+1} [{status}]: Start={detector.starttime[i]:.4f}, Peak={detector.peaktime[i]:.4f}, End={detector.endtime[i]:.4f}, Energy={detector.energy[i]:.2e}")
