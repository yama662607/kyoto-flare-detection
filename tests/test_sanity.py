from pathlib import Path
import importlib
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_modules_importable():
    modules = [
        "src.base_flare_detector",
        "src.flarepy_EK_Dra",
        "src.flarepy_DS_Tuc_A",
        "src.flarepy_V889_Her",
    ]
    for name in modules:
        importlib.import_module(name)
