"""Put src/ on sys.path and make `seasonal_smoothing` importable for every test module.

edineof.py and composite.py import `seasonal_smoothing` from a sibling cloud_mixture_model/src
that this repo does not have; iterative_filter's bridge finds it in the coastal_sst_data
checkout. Importing it here, before collection, covers test modules that import edineof first.
"""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import iterative_filter  # noqa: E402,F401
