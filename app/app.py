"""
Minimal wrapper entrypoint for Streamlit hosting.

This file imports and executes the root `app.py` so deployments using
`app/app.py` get the canonical app. If import fails, it renders a simple
error page with the traceback so the site isn't blank.
"""
from pathlib import Path
import sys
import importlib
import traceback

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    importlib.import_module("app")
except Exception:
    # Print to stdout/stderr so hosting logs capture the traceback
    traceback.print_exc()
    try:
        import streamlit as st
        st.set_page_config(page_title="PitchSequence — import error")
        st.title("PitchSequence — Import error (wrapper)")
        st.error("Failed to import the root app. Check the deployment logs for traceback.")
        st.text(traceback.format_exc())
    except Exception:
        # If Streamlit isn't available in this context, just exit after printing
        pass
    # Re-raise so the hosting logs clearly show the import failure
    raise


