import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from attack_recommender import load_data, recommend_sequence, get_archetype, get_next_pitch_candidates
from predict import recommend_next_pitch, MODEL_PATH

st.set_page_config(page_title="PitchSequence", layout="centered")
st.title("PitchSequence — Revamped Pitch Sequence Recommender")


# Safe experimental rerun: some deployed Streamlit versions don't expose
# st.experimental_rerun (raises AttributeError). Use this helper to
# call it when available; otherwise it's a no-op so the app won't crash.
def _safe_rerun():
    try:
        if hasattr(st, "experimental_rerun") and callable(st.experimental_rerun):
            st.experimental_rerun()
    except Exception:
        # swallow any issues with rerun on older/newer Streamlit builds
        return

# Load data
ARCH_PATH = Path("data/player_archetypes.csv")
ARS_PATH = Path("pitcher_assets/pitcher_arsenals.csv")
arche_df, ars_df = load_data(ARCH_PATH, ARS_PATH)

# Load name -> id mapping files (prefer data/raw versions if present)
BATTER_MAP_PATH = Path("data/raw/unique_batters_with_names.csv")
PITCHER_MAP_PATH = Path("data/raw/unique_pitchers_with_names.csv")

def _load_name_map(p: Path, id_col: str = None):
    if not p.exists():
        return {}, []
    try:
        df = pd.read_csv(p)
        # expect first two columns to be id,name
        if df.shape[1] >= 2:
            id_col_name = df.columns[0]
            name_col = df.columns[1]
            mapping = {str(r[name_col]): str(r[id_col_name]) for _, r in df.iterrows()}
            names = sorted(list(mapping.keys()))
            return mapping, names
    except Exception:
        pass
    return {}, []

batter_name_to_id, batter_names = _load_name_map(BATTER_MAP_PATH)
pitcher_name_to_id, pitcher_names = _load_name_map(PITCHER_MAP_PATH)

st.markdown("Enter a batter name and a pitcher name (names only). The app will map names to ids behind the scenes and use the batter's archetype from the archetypes CSV to recommend a pitch sequence aimed at minimizing runs.")
st.session_state.setdefault("atbat_active", False)

if not st.session_state.get("atbat_active"):
    with st.form(key='start_atbat'):
        col1, col2 = st.columns([2, 1])
        with col1:
            # Batter search + filtered selectbox (type to filter)
            if batter_names:
                batter_search = st.text_input("Search batter", value="")
                # filter by substring match (case-insensitive)
                if batter_search:
                    filtered = [n for n in batter_names if batter_search.lower() in n.lower()]
                else:
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

                    # Import the root-level app module (this will execute it when Streamlit
                    # launches this file). We import with a different name to avoid clobbering
                    # the module namespace if run interactively.
                    import importlib

                    try:
                        importlib.import_module('app')
                    except Exception as e:
                        # If importing fails, show a helpful message so deployment logs reveal
                        # the root cause quickly.
                        print(f"Failed to import root app module: {e}")
        st.markdown("Select which bases are occupied (visual triangle).")

        b1, b2, b3 = st.columns(3)

        with b1:
