"""
Light wrapper for the root-level Streamlit app.

Some hosting environments (and previous deployments) run the app
from the path `app/app.py`. To keep a single source of truth and avoid
stale deployments using an out-of-sync file, this small wrapper imports
and executes the root `app.py` module so both entrypoints behave the same.

This file intentionally keeps logic minimal so the real app lives in
`app.py` at the repository root.
"""
import sys
from pathlib import Path

# Add repository root to path so importing the root-level app works
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib

try:
    importlib.import_module('app')
except Exception as e:
    # If importing fails, print to stdout so deployment logs show the error
    # and Streamlit fails loudly with useful context.
    print(f"Failed to import root app module: {e}")

