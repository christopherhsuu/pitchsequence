"""Minimal wrapper so hosting configured with main module `app/app.py`
imports and runs the canonical root `app.py`.

This file intentionally keeps logic minimal to avoid parse-time issues on
managed hosts. If the import fails we print the traceback and, when
Streamlit is available, render a small error page so the deployed site isn't
blank.
"""
from pathlib import Path
import sys
import importlib
import traceback

# Ensure repository root is on sys.path so `import app` resolves to the
# canonical `app.py` at the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    # Importing the root module 'app' will execute the main Streamlit app.
    importlib.import_module("app")
except Exception:
    # Print traceback so hosting logs capture the error.
    traceback.print_exc()
    # Try to render a minimal Streamlit error page so the site surfaces
    # useful information instead of a blank page.
    try:
        import streamlit as st
        st.set_page_config(page_title="PitchSequence — import error")
        st.title("PitchSequence — Import error (wrapper)")
        st.error("Failed to import the root app. Check the deployment logs for traceback.")
        st.text(traceback.format_exc())
    except Exception:
        # If Streamlit isn't importable in this context, silently continue
        # after printing the traceback so hosting logs still include the error.
        pass
    # Re-raise so hosting systems record the failure state.
    raise


