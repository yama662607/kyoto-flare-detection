import os
import sys
from pathlib import Path

# Mocking the notebook environment where the file is in notebooks/
# We are running this script in c:\Users\81803\Documents\フレア\kyoto-flare-detection\notebooks
# So Path().resolve() should be the notebooks dir.

NOTEBOOK_DIR = Path().resolve()
PROJECT_ROOT = NOTEBOOK_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print('Project:', PROJECT_ROOT)

print("Attempting imports...")
try:
    from src import flarepy_EK_Dra as ek_module
    from src import flarepy_V889_Her as v889_module
    from src import flarepy_DS_Tuc_A as ds_module
    print("Successfully imported star modules from src.")
    
    print("Checking Classes:")
    print(f"EK Dra Class: {ek_module.FlareDetector_EK_Dra.__name__}")
    print(f"V889 Class: {v889_module.FlareDetector_V889_Her.__name__}")
    print(f"DS Tuc Class: {ds_module.FlareDetector_DS_Tuc_A.__name__}")

except ImportError as e:
    print(f"Error importing modules from src: {e}")
    exit(1)
except AttributeError as e:
    print(f"Error accessing class in module: {e}")
    exit(1)

print("Verification Successful.")
