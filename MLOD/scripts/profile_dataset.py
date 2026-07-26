#!/usr/bin/env python3
"""Entry point placeholder for the dataset profiling utility.

Place the previously generated dataset_profile.py implementation here or call it
from this entry point. It is intentionally separated from the training package
because profiling does not import PyTorch detection components.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROFILER = PROJECT_ROOT / "utils" / "dataset_profile.py"

if __name__ == "__main__":
    if not PROFILER.exists():
        raise FileNotFoundError(
            "Copy dataset_profile.py to utils/dataset_profile.py first."
        )
    import runpy
    sys.argv[0] = str(PROFILER)
    runpy.run_path(str(PROFILER), run_name="__main__")
