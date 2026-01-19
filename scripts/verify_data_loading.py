import os
import sys
from pathlib import Path

# Setup Path
NOTEBOOK_DIR = Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Imports
try:
    from src import flarepy_EK_Dra as ek_module
    from src import flarepy_V889_Her as v889_module
    from src import flarepy_DS_Tuc_A as ds_module
    print("Modules imported successfully.")
except ImportError as e:
    print(f"Import Error: {e}")
    exit(1)

# STAR_INFO
STAR_INFO = {
    "EKDra":    {"class": ek_module.FlareDetector_EK_Dra,   "label": "EK Dra"},
    "V889 Her": {"class": v889_module.FlareDetector_V889_Her, "label": "V889 Her"},
    "DS Tuc":   {"class": ds_module.FlareDetector_DS_Tuc_A,   "label": "DS Tuc"},
}

BASE_STARS_FOLDER = "C:/Users/81803/Documents/フレア/Kyoto_Student_Flare_Project/all_stars/"
all_stars_detectors = {}

print(f"Checking Base Folder: {BASE_STARS_FOLDER}")
if not os.path.exists(BASE_STARS_FOLDER):
    print("Base folder NOT FOUND.")
else:
    print("Base folder exists. Listing content:")
    folders = os.listdir(BASE_STARS_FOLDER)
    print(folders)
    
    for star_folder_name in folders:
        star_folder_path = os.path.join(BASE_STARS_FOLDER, star_folder_name)
        if not os.path.isdir(star_folder_path):
            continue
            
        if star_folder_name not in STAR_INFO:
            print(f"Skipping {star_folder_name} (not in STAR_INFO)")
            continue
            
        print(f"Processing {star_folder_name}...")
        TargetFlareDetectorClass = STAR_INFO[star_folder_name]["class"]
        
        # Check files
        files = [f for f in os.listdir(star_folder_path) if f.endswith(".fits")]
        print(f"  Found {len(files)} fits files.")
        
        full_paths = [os.path.join(star_folder_path, f) for f in files]
        
        # Test loading one file to be sure
        if full_paths:
            try:
                print(f"  Attempting to load one file: {files[0]}")
                # Use faster params for test
                detector = TargetFlareDetectorClass(
                    file=full_paths[0],
                    process_data=True,
                    ene_thres_high=1e39
                )
                print("  Success loading file.")
                if hasattr(detector, 'tessBJD') and detector.tessBJD is not None:
                     print("  Data attribute 'tessBJD' is present.")
                else:
                     print("  WARNING: 'tessBJD' is missing or empty.")
            except Exception as e:
                print(f"  Error loading file: {e}")
        else:
            print("  No fits files to test.")

print("Verification script finished.")
