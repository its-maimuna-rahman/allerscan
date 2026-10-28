"""
AllerScan Core Package
Provides allergen knowledge base, NLP parser, model severity prediction,
and photo scanning pipelines.
"""
from pathlib import Path
import sys

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
