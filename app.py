"""Root Application Entry Point for Streamlit."""

import sys
import runpy
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
MAIN_SCRIPT = ROOT_DIR / "app" / "main.py"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if __name__ == "__main__":
    runpy.run_path(str(MAIN_SCRIPT), run_name="__main__")
