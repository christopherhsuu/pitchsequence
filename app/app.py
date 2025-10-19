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

"""
Fallback minimal app — if the full app fails to import or the host keeps
running a stale/corrupted entrypoint, show a helpful fallback UI with
diagnostics so you can see the deployed commit and basic environment info.

This file intentionally avoids importing the project so it should always
render, even when the main app fails.
"""

import streamlit as st
import subprocess
import os

st.set_page_config(page_title="PitchSequence (fallback)")
st.title("PitchSequence — Minimal fallback UI")

# show best-effort commit hash
short_hash = None
try:
    short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
except Exception:
    pass

if short_hash:
    st.caption(f"commit: {short_hash}")
else:
    st.caption("commit: unknown")

st.info("The full app import failed or the deployment is running a stale file. This is a minimal fallback so the site isn't blank.")

st.subheader("Diagnostics")
st.write("Repository root files (top-level):")
try:
    root = Path(__file__).resolve().parents[1]
    files = sorted([p.name for p in root.iterdir() if p.exists()])
    st.write(files)
except Exception as e:
    st.write(str(e))

st.write("If you see this, please redeploy the app after verifying the repository's `main` branch is up-to-date. I can also add more diagnostics if helpful.")

