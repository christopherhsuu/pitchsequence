import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
# Ensure src is importable when running the app from repository root
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

IMPORT_ERROR = None
try:
    # Import new recommender system
    from predict import PitchRecommender, recommend_next_pitch

    # Try to import old modules for backwards compatibility (optional)
    try:
        from attack_recommender import load_data, recommend_sequence, get_archetype, get_next_pitch_candidates, map_pitch_to_location, load_batter_pitchtype_stats, get_batter_pitchtype_stats
    except ImportError:
        # Old modules not available - define fallbacks
        def load_data(archetypes_path, arsenals_path):
            a = pd.DataFrame(columns=['batter','cluster','label']) if not Path(archetypes_path).exists() else pd.read_csv(archetypes_path)
            ars = pd.DataFrame(columns=['pitcher','pitch_type']) if not Path(arsenals_path).exists() else pd.read_csv(arsenals_path)
            return a, ars

        def get_archetype(arche_df, batter_id):
            return "unknown"

        def get_pitcher_arsenal(ars_df, pitcher_id):
            """Helper to get pitcher arsenal from new CSV format."""
            if not isinstance(ars_df, pd.DataFrame):
                return []
            try:
                pitcher_row = ars_df[ars_df["pitcher"].astype(str) == str(pitcher_id)]
                if pitcher_row.empty:
                    return []
                # New format has 'pitch_types' column with comma-separated values
                if 'pitch_types' in pitcher_row.columns:
                    pitch_types_str = pitcher_row.iloc[0]["pitch_types"]
                    if pd.notna(pitch_types_str):
                        return [pt.strip() for pt in str(pitch_types_str).split(',')]
                # Fallback: old format with individual 'pitch_type' rows
                elif 'pitch_type' in ars_df.columns:
                    return pitcher_row["pitch_type"].astype(str).unique().tolist()
            except Exception:
                pass
            return []

        def get_next_pitch_candidates(arche_df, ars_df, batter, pitcher, count, situation, top_n=None):
            p_list = get_pitcher_arsenal(ars_df, pitcher)
            if not p_list:
                p_list = ["FF","SL","CU"]
            pct = round(100.0/len(p_list),1) if p_list else 0
            return [{"pitch": p, "pct": pct, "location": "middle"} for p in p_list]

        def map_pitch_to_location(pitch_type, archetype, situation):
            m = {"FF":"up-in","SL":"low-away","CU":"low-middle"}
            return m.get(str(pitch_type).upper(), "middle")

        def load_batter_pitchtype_stats(path="data/processed/features.parquet"):
            return None

        def get_batter_pitchtype_stats(stats_df, batter_id, pitch_type, p_throws="R"):
            return 0.0, 0.0, 0.0

    # Define missing functions if not imported from old modules
    if 'get_cluster_features' not in dir():
        def get_cluster_features(cluster_id):
            return {}

    if 'adjust_pitch_recommendation' not in dir():
        def adjust_pitch_recommendation(prob_dict, cluster_feats):
            return prob_dict

except Exception as e:
    # Don't raise here — show a helpful message in the UI so deployment isn't a blank page.
    IMPORT_ERROR = e

# Model and data paths
MODEL_PATH = Path("artifacts/pitch_model.pkl")
ARTIFACTS_PATH = Path("artifacts")

# Initialize PitchRecommender
try:
    pitch_recommender = PitchRecommender()
    print("PitchRecommender initialized successfully")
except Exception as e:
    print(f"Warning: Could not initialize PitchRecommender: {e}")
    pitch_recommender = None

# Pitch type color mapping for visual consistency
PITCH_COLORS = {
    "FF": "#ef4444",  # Red - Fastball
    "SI": "#f97316",  # Orange - Sinker
    "FC": "#f59e0b",  # Amber - Cutter
    "SL": "#3b82f6",  # Blue - Slider
    "CU": "#10b981",  # Green - Curveball
    "CH": "#ec4899",  # Pink - Changeup
    "FS": "#8b5cf6",  # Purple - Splitter
    "KC": "#14b8a6",  # Teal - Knuckle Curve
    "ST": "#6366f1",  # Indigo - Sweeper
    "SV": "#06b6d4",  # Cyan - Slurve
}

def get_pitch_color(pitch_type):
    """Return color for pitch type, with fallback to gray."""
    return PITCH_COLORS.get(pitch_type, "#6b7280")

def render_baseball_diamond(on_1b, on_2b, on_3b, outs=0):
    """Render an SVG baseball diamond showing base runners."""
    first_base_color = "#fbbf24" if on_1b else "#ffffff"
    second_base_color = "#fbbf24" if on_2b else "#ffffff"
    third_base_color = "#fbbf24" if on_3b else "#ffffff"

    runners_svg = ""
    if on_1b:
        runners_svg += "<circle cx='85' cy='50' r='5' fill='#dc2626' stroke='#fff' stroke-width='1'/>"
    if on_2b:
        runners_svg += "<circle cx='50' cy='15' r='5' fill='#dc2626' stroke='#fff' stroke-width='1'/>"
    if on_3b:
        runners_svg += "<circle cx='15' cy='50' r='5' fill='#dc2626' stroke='#fff' stroke-width='1'/>"

    diamond_svg = f'<div style="text-align: center; margin: 20px 0;"><svg width="240" height="240" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><path d="M 50 10 L 90 50 L 50 90 L 10 50 Z" fill="#2d5016" stroke="#4ade80" stroke-width="2"/><path d="M 50 96 L 46 92 L 46 88 L 54 88 L 54 92 Z" fill="#ffffff" stroke="#000" stroke-width="1.5"/><rect x="80" y="45" width="10" height="10" fill="{first_base_color}" stroke="#000" stroke-width="1.5" transform="rotate(45 85 50)"/><rect x="45" y="10" width="10" height="10" fill="{second_base_color}" stroke="#000" stroke-width="1.5" transform="rotate(45 50 15)"/><rect x="10" y="45" width="10" height="10" fill="{third_base_color}" stroke="#000" stroke-width="1.5" transform="rotate(45 15 50)"/>{runners_svg}</svg></div>'
    return diamond_svg

def render_count_display(count_str):
    """Render a scoreboard-style count display."""
    try:
        balls, strikes = count_str.split('-')
        balls = int(balls)
        strikes = int(strikes)
    except:
        balls, strikes = 0, 0

    balls_filled = "".join([f'<circle cx="{20 + i*20}" cy="50" r="8" fill="#22c55e" stroke="#fff" stroke-width="2"/>' for i in range(balls)])
    balls_empty = "".join([f'<circle cx="{20 + i*20}" cy="50" r="8" fill="#374151" stroke="#6b7280" stroke-width="2"/>' for i in range(balls, 4)])
    strikes_filled = "".join([f'<circle cx="{120 + i*20}" cy="50" r="8" fill="#ef4444" stroke="#fff" stroke-width="2"/>' for i in range(strikes)])
    strikes_empty = "".join([f'<circle cx="{120 + i*20}" cy="50" r="8" fill="#374151" stroke="#6b7280" stroke-width="2"/>' for i in range(strikes, 3)])

    count_svg = f'<div style="text-align: center;"><svg width="200" height="100" viewBox="0 0 200 100" xmlns="http://www.w3.org/2000/svg"><rect width="200" height="100" fill="#1f2937" rx="8"/><text x="50" y="25" font-size="12" fill="#9ca3af" text-anchor="middle" font-weight="bold">BALLS</text>{balls_filled}{balls_empty}<text x="150" y="25" font-size="12" fill="#9ca3af" text-anchor="middle" font-weight="bold">STRIKES</text>{strikes_filled}{strikes_empty}<text x="100" y="85" font-size="18" fill="#ffffff" text-anchor="middle" font-weight="bold">{balls}-{strikes}</text></svg></div>'
    return count_svg

def render_pitch_history(history):
    """Render a timeline of pitch history using Streamlit components."""
    if not history:
        st.markdown("<p style='color: #6b7280; font-style: italic;'>No pitches thrown yet</p>", unsafe_allow_html=True)
        return

    cols = st.columns(min(len(history), 6))
    for i, h in enumerate(history):
        pitch = h.get('pitch', '?')
        outcome = h.get('outcome', '?')
        count_before = h.get('count_before', '?')
        color = get_pitch_color(pitch)

        # Outcome indicator
        if outcome == "Ball":
            outcome_color = "#22c55e"
            outcome_icon = "B"
        elif "Strike" in outcome:
            outcome_color = "#ef4444"
            outcome_icon = "S"
        elif "Foul" in outcome:
            outcome_color = "#f97316"
            outcome_icon = "F"
        elif "In play" in outcome:
            outcome_color = "#3b82f6"
            outcome_icon = "IP"
        else:
            outcome_color = "#6b7280"
            outcome_icon = "?"

        with cols[i % 6]:
            card_html = f'''
            <div style='border: 2px solid {color}; border-radius: 8px; padding: 8px 12px; background: #f9fafb; min-width: 100px; margin-bottom: 8px;'>
                <div style='font-weight: bold; color: {color}; font-size: 14px;'>{pitch}</div>
                <div style='font-size: 11px; color: #6b7280;'>Count: {count_before}</div>
                <div style='background: {outcome_color}; color: white; border-radius: 4px; padding: 2px 6px; font-size: 10px; margin-top: 4px; text-align: center;'>{outcome_icon}: {outcome}</div>
            </div>
            '''
            st.markdown(card_html, unsafe_allow_html=True)

st.set_page_config(page_title="PitchSequence", layout="centered")
st.title("PitchSequence \u2014 Pitch Sequence Recommender")
# Prominent deployed-commit banner and artifact presence check (helps hosted debugging)
try:
    import subprocess, os
    short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    model_path = os.path.exists("artifacts/runvalue_model.pkl")
    ars_path = os.path.exists("pitcher_assets/pitcher_arsenals.csv")
    # Deployment info is shown in the sidebar for less intrusive UI; remove top banner
    try:
        st.sidebar.markdown(f"**Deployment**: commit `{short_hash}` — model: {model_path} — arsenals: {ars_path}")
    except Exception:
        # fallback to subtle caption if sidebar isn't available
        st.caption(f"commit: {short_hash}  model: {model_path}  arsenals: {ars_path}")

    # Verbose diagnostics: why are mapping CSVs not being read on the host?
    try:
        import os as _os
        import traceback as _tb
        from pathlib import Path as _P

        _diag = {}
        _diag['cwd'] = str(_os.getcwd())
        _diag['cwd_listing'] = []
        try:
            _diag['cwd_listing'] = sorted(_os.listdir('.'))
        except Exception:
            _diag['cwd_listing'] = f'error: {_tb.format_exc()}'

        # candidate paths
        _bmap = _P("data/raw/unique_batters_with_names.csv")
        _pmap = _P("data/raw/unique_pitchers_with_names.csv")
        _candidates = [str(_bmap), str(_P('data') / _bmap.name), str(_P(_bmap.name))]
        _diag['batter_candidates'] = _candidates
        _diag['batter_exists'] = {c: _P(c).exists() for c in _candidates}
        _diag['pitcher_candidates'] = [str(_pmap), str(_P('data') / _pmap.name), str(_P(_pmap.name))]
        _diag['pitcher_exists'] = {c: _P(c).exists() for c in _diag['pitcher_candidates']}

        # file sizes where present
        def _size(pth):
            try:
                return int(_P(pth).stat().st_size)
            except Exception:
                return None

        _diag['batter_sizes'] = {c: _size(c) for c in _diag['batter_candidates']}
        _diag['pitcher_sizes'] = {c: _size(c) for c in _diag['pitcher_candidates']}

        # try reading with pandas and capture errors
        _diag['batter_read'] = None
        for c in _diag['batter_candidates']:
            try:
                if _P(c).exists():
                    df = pd.read_csv(c)
                    _diag['batter_read'] = {'path': c, 'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
                    break
            except Exception:
                _diag.setdefault('batter_read_errors', []).append({'path': c, 'err': _tb.format_exc()})

        _diag['pitcher_read'] = None
        for c in _diag['pitcher_candidates']:
            try:
                if _P(c).exists():
                    df = pd.read_csv(c)
                    _diag['pitcher_read'] = {'path': c, 'rows': int(df.shape[0]), 'cols': int(df.shape[1])}
                    break
            except Exception:
                _diag.setdefault('pitcher_read_errors', []).append({'path': c, 'err': _tb.format_exc()})

        # try to fetch raw from GitHub (if network available)
        try:
            import requests as _req
            _gh_base = globals().get('GITHUB_RAW_BASE', "https://raw.githubusercontent.com/christopherhsuu/pitchsequence/main/")
            _gh_b = _gh_base + str(_P('data/raw') / _bmap.name)
            _gh_p = _gh_base + str(_P('data/raw') / _pmap.name)
            _diag['github_batter_fetch'] = None
            _diag['github_pitcher_fetch'] = None
            try:
                r = _req.get(_gh_b, timeout=6)
                _diag['github_batter_fetch'] = {'status': r.status_code, 'len': len(r.content)}
            except Exception:
                _diag['github_batter_fetch'] = {'error': _tb.format_exc()}
            try:
                r = _req.get(_gh_p, timeout=6)
                _diag['github_pitcher_fetch'] = {'status': r.status_code, 'len': len(r.content)}
            except Exception:
                _diag['github_pitcher_fetch'] = {'error': _tb.format_exc()}
        except Exception:
            _diag['github_fetch_error'] = _tb.format_exc()

        # include _MAP_SOURCES if available
        try:
            _diag['_MAP_SOURCES'] = globals().get('_MAP_SOURCES', {}).copy()
        except Exception:
            _diag['_MAP_SOURCES'] = None

        # check run expectancy map
        try:
            _re_map = globals().get('_RE_MAP', {})
            _diag['_RE_MAP_loaded'] = bool(_re_map)
            _diag['_RE_MAP_size'] = len(_re_map) if _re_map else 0
            _diag['_RE_MAP_sample'] = dict(list(_re_map.items())[:3]) if _re_map else {}
        except Exception:
            _diag['_RE_MAP_error'] = _tb.format_exc()

        # show a compact version in the sidebar and a full expander
        try:
            st.sidebar.markdown(f"**Mappings diagnostics:** batter_found={bool(_diag.get('batter_read'))} pitcher_found={bool(_diag.get('pitcher_read'))} RE_map={_diag.get('_RE_MAP_size', 0)} entries")
            with st.sidebar.expander('Mappings diagnostics (details)', expanded=False):
                st.write(_diag)
        except Exception:
            # best-effort only
            pass
    except Exception:
        # don't fail the app for diagnostics
        pass
except Exception:
    # best-effort only — don't fail the app for banner rendering
    pass

# show current commit (best-effort) for deployed visibility
try:
    import subprocess
    short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    # show prominently in the sidebar and also as a small caption so it's easy to verify the deployed commit
    try:
        st.sidebar.markdown(f"**commit:** `{short_hash}`  \n**branch:** {branch_name}")
        st.info(f"Running commit: {short_hash} (branch: {branch_name})")
    except Exception:
        # fallback to caption when sidebar/info unavailable
        st.caption(f"commit: {short_hash}  branch: {branch_name}")
except Exception:
    # ignore if git not available in the runtime
    pass

# If importing project modules failed, surface the error and stop the app
if IMPORT_ERROR is not None:
    # Don't crash the app in hosted environments — surface the import error but provide
    # safe fallback implementations so the UI can still run in a degraded mode.
    import traceback
    st.warning("Project modules failed to import; running in degraded mode. See sidebar for details.")
    with st.sidebar.expander("Import error (details)"):
        st.text(str(IMPORT_ERROR))
        st.text(traceback.format_exc())

    # Provide minimal fallback implementations used by the app so the UI can operate.
    def load_data(archetypes_path, arsenals_path):
        # try to read if files exist, else return empty DataFrames
        try:
            a = pd.read_csv(archetypes_path) if Path(archetypes_path).exists() else pd.DataFrame(columns=['batter','cluster','label'])
        except Exception:
            a = pd.DataFrame(columns=['batter','cluster','label'])
        try:
            ars = pd.read_csv(arsenals_path) if Path(arsenals_path).exists() else pd.DataFrame(columns=['pitcher','pitch_type'])
        except Exception:
            ars = pd.DataFrame(columns=['pitcher','pitch_type'])
        return a, ars

    def recommend_sequence(*args, **kwargs):
        return {"recommended_sequence": [], "confidence": 0.0, "strategy_notes": "degraded-mode"}

    def get_archetype(arche_df, batter_id):
        return "unknown"

    def get_pitcher_arsenal(ars_df, pitcher_id):
        """Helper to get pitcher arsenal from new CSV format."""
        if not isinstance(ars_df, pd.DataFrame):
            return []
        try:
            pitcher_row = ars_df[ars_df["pitcher"].astype(str) == str(pitcher_id)]
            if pitcher_row.empty:
                return []
            # New format has 'pitch_types' column with comma-separated values
            if 'pitch_types' in pitcher_row.columns:
                pitch_types_str = pitcher_row.iloc[0]["pitch_types"]
                if pd.notna(pitch_types_str):
                    return [pt.strip() for pt in str(pitch_types_str).split(',')]
            # Fallback: old format with individual 'pitch_type' rows
            elif 'pitch_type' in ars_df.columns:
                return pitcher_row["pitch_type"].astype(str).unique().tolist()
        except Exception:
            pass
        return []

    def get_next_pitch_candidates(arche_df, ars_df, batter, pitcher, count, situation, top_n=None):
        # simple fallback candidate list
        p_list = get_pitcher_arsenal(ars_df, pitcher)
        if not p_list:
            p_list = ["FF","SL","CU"]
        pct = round(100.0/len(p_list),1) if p_list else 0
        return [{"pitch": p, "pct": pct, "location": "middle"} for p in p_list]

    def map_pitch_to_location(pitch_type, archetype, situation):
        m = {"FF":"up-in","SL":"low-away","CU":"low-middle"}
        return m.get(str(pitch_type).upper(), "middle")

    def load_batter_pitchtype_stats(path="data/processed/features.parquet"):
        return None

    def get_batter_pitchtype_stats(stats_df, batter_id, pitch_type, p_throws="R"):
        return 0.0, 0.0, 0.0

    def get_cluster_features(cluster_id):
        return {}

    def adjust_pitch_recommendation(prob_dict, cluster_feats):
        return prob_dict


# Safe experimental rerun helper: prevents AttributeError when deployed
# to environments with older/newer Streamlit versions that don't expose
# the experimental API. It's intentionally a no-op when unavailable.
def _safe_rerun():
    try:
        if hasattr(st, "experimental_rerun") and callable(st.experimental_rerun):
            st.experimental_rerun()
    except Exception:
        return

# Load data - use new profile-based system
# Try to load old archetype data if available, otherwise use empty DataFrames
ARCH_PATH = Path("data/player_archetypes.csv")
ARS_PATH = Path("data/processed/pitcher_arsenals.csv")  # Updated path for new system

try:
    if 'load_data' in dir():
        arche_df, ars_df = load_data(ARCH_PATH, ARS_PATH)
    else:
        # Fallback: load directly
        arche_df = pd.read_csv(ARCH_PATH) if ARCH_PATH.exists() else pd.DataFrame(columns=['batter','cluster','label'])
        ars_df = pd.read_csv(ARS_PATH) if ARS_PATH.exists() else pd.DataFrame(columns=['pitcher','pitch_type'])
except Exception:
    # Use empty DataFrames if files don't exist
    arche_df = pd.DataFrame(columns=['batter','cluster','label'])
    ars_df = pd.read_csv(ARS_PATH) if ARS_PATH.exists() else pd.DataFrame(columns=['pitcher','pitch_type'])

# load batter vs pitch_type matchup stats (if processed features exist)
try:
    if 'load_batter_pitchtype_stats' in dir():
        _batter_pitch_stats_df = load_batter_pitchtype_stats()
    else:
        _batter_pitch_stats_df = None
except Exception:
    _batter_pitch_stats_df = None


def _load_run_expectancy():
    """Load run expectancy matrix from CSV file.

    Tries multiple candidate paths in order:
    1. static/run_expectancy_24.csv
    2. run_expectancy_24.csv (top-level)
    3. GitHub raw URL as fallback
    """
    candidates = [
        Path("static/run_expectancy_24.csv"),
        Path("run_expectancy_24.csv"),
    ]

    for p in candidates:
        if p.exists():
            try:
                # Read bases_state as string to preserve leading zeros
                df = pd.read_csv(p, dtype={'bases_state': str})
                re_map = {(int(r.outs), str(r.bases_state)): float(r.run_expectancy) for r in df.itertuples()}
                if re_map:  # Only return if we got data
                    return re_map
            except Exception:
                continue

    # Last resort: try GitHub raw URL
    try:
        url = "https://raw.githubusercontent.com/christopherhsuu/pitchsequence/main/static/run_expectancy_24.csv"
        df = pd.read_csv(url, dtype={'bases_state': str})
        re_map = {(int(r.outs), str(r.bases_state)): float(r.run_expectancy) for r in df.itertuples()}
        if re_map:
            return re_map
    except Exception:
        pass

    # If all else fails, return empty dict
    return {}


_RE_MAP = _load_run_expectancy()


def _loc_to_svg_coords(loc: str):
    """Map a human location string (e.g., 'low-away', 'up-in') to SVG cx, cy coordinates.

    Returns coordinates in the same 0-100 viewBox used for the small strike-zone SVGs.
    """
    # default center
    x = 50
    y = 50
    if not loc:
        return x, y
    s = str(loc).lower()
    # horizontal
    if 'in' in s:
        x = 65
    elif 'out' in s or 'away' in s:
        x = 35
    elif 'edge' in s:
        # edge can be a bit wider
        x = 30
    elif 'middle' in s:
        x = 50

    # vertical
    if 'low' in s:
        y = 70
    elif 'up' in s or 'high' in s:
        y = 30
    elif 'middle' in s or 'middle-strike' in s:
        y = 50

    # small adjustments for more descriptive tokens
    if 'arm' in s:
        x = 60
    if 'edge-low-away' in s:
        x, y = 28, 72

    return x, y


def _prob_to_visuals(pct: float):
    """Map a probability percentage to circle size and color intensity.

    Returns (r, color_hex).
    """
    try:
        p = float(pct)
    except Exception:
        p = 0.0
    # radius between 5 and 12
    r = 5 + (p / 100.0) * 7
    # color: interpolate from light red to dark red
    # simple approach: more prob -> darker (lower green)
    green = int(max(80, 200 - (p / 100.0) * 160))
    color = f"#e53e3e" if p < 1 else f"rgb(229,{green},62)"
    return r, color


def _plate_xy_to_svg_coords(plate_x: float, plate_z: float):
    """Convert plate coordinates to SVG coords (0-100 viewBox).

    Args:
        plate_x: Horizontal position (-1.5 to +1.5 feet, 0 = middle)
        plate_z: Vertical position (1.5 to 3.5 feet, height from ground)

    Returns:
        (cx, cy) in 0-100 SVG viewBox coordinates
    """
    # Strike zone boundaries in the SVG (viewBox 0-100):
    # x: 15-85 (70 units wide)
    # y: 8-92 (84 units tall)

    # Map plate_x (-1.5 to +1.5) to SVG x (85 to 15) - reversed because negative is inside (right in catcher's view)
    # Note: Reverse the mapping so negative plate_x (inside to RHH) appears on the right (higher SVG x)
    cx = 50 - (plate_x / 1.5) * 35  # Center at 50, range 15-85

    # Map plate_z (1.5 to 3.5) to SVG y (92 to 8) - reversed because higher plate_z = lower SVG y
    cz = 92 - ((plate_z - 1.5) / 2.0) * 84  # Bottom at 92, top at 8

    # Clamp to viewBox bounds
    cx = max(15, min(85, cx))
    cz = max(8, min(92, cz))

    return cx, cz


def _svg_coords_to_plate_xy(cx: float, cy: float):
    """Convert SVG coords (0-100 viewBox) to approximate plate_x and plate_z.

    - plate_x roughly maps from -1.5 (far-away) to +1.5 (far-in)
    - plate_z roughly maps from 0.5 to 4.0 (low to high)
    """
    # normalize
    nx = (cx - 50) / 50.0  # -1..+1
    plate_x = nx * 1.5
    # y: 0..100 -> high to low
    nz = (cy - 30) / 70.0  # approximate mapping
    plate_z = max(0.5, min(4.0, 4.0 - nz * 3.5))
    return plate_x, plate_z

# Load name -> id mapping files (prefer data/raw versions if present)
BATTER_MAP_PATH = Path("data/raw/unique_batters_with_names.csv")
PITCHER_MAP_PATH = Path("data/raw/unique_pitchers_with_names.csv")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/christopherhsuu/pitchsequence/main/"

# record where each mapping was loaded from (for diagnostics)
_MAP_SOURCES = {}

def _load_name_map(p: Path, id_col: str = None):
    """Load a mapping CSV from multiple candidate locations.

    Tries these in order:
      1. the provided path `p`
      2. `data/` + basename(p)
      3. top-level basename(p)
      4. the GitHub raw URL for the file

    Returns (mapping_dict, sorted_names_list). Records the successful source
    in _MAP_SOURCES keyed by the basename.
    """
    name = p.name
    candidates = [p, Path("data") / name, Path(name)]

    for c in candidates:
        try:
            if c.exists():
                df = pd.read_csv(c)
                if df.shape[1] >= 2:
                    id_col_name = df.columns[0]
                    name_col = df.columns[1]
                    mapping = {str(r[name_col]): str(r[id_col_name]) for _, r in df.iterrows()}
                    names = sorted(list(mapping.keys()))
                    _MAP_SOURCES[name] = str(c)
                    return mapping, names
        except Exception:
            # try next candidate
            continue

    # Last resort: try to load from GitHub raw URL
    try:
        url = GITHUB_RAW_BASE + str(Path("data/raw") / name)
        df = pd.read_csv(url)
        if df.shape[1] >= 2:
            id_col_name = df.columns[0]
            name_col = df.columns[1]
            mapping = {str(r[name_col]): str(r[id_col_name]) for _, r in df.iterrows()}
            names = sorted(list(mapping.keys()))
            _MAP_SOURCES[name] = url
            return mapping, names
    except Exception:
        pass

    # nothing worked
    _MAP_SOURCES[name] = None
    return {}, []

batter_name_to_id, batter_names = _load_name_map(BATTER_MAP_PATH)
pitcher_name_to_id, pitcher_names = _load_name_map(PITCHER_MAP_PATH)

st.markdown("Set up the game situation and start the at-bat. Click bases to toggle runners, click outs circles to change count.")
st.session_state.setdefault("atbat_active", False)

# Initialize setup state
st.session_state.setdefault("setup_on_1b", False)
st.session_state.setdefault("setup_on_2b", False)
st.session_state.setdefault("setup_on_3b", False)
st.session_state.setdefault("setup_outs", 0)

# ensure submit exists even if form isn't rendered
submit = False

if not st.session_state.get("atbat_active"):
    # Player selection
    col1, col2 = st.columns(2)
    with col1:
        if batter_names:
            batter_name = st.selectbox("Batter", options=batter_names, index=0)
        else:
            batter_name = st.text_input("Batter", value="", placeholder="Type batter name")

    with col2:
        if pitcher_names:
            pitcher_name = st.selectbox("Pitcher", options=pitcher_names, index=0)
        else:
            pitcher_name = st.text_input("Pitcher", value="", placeholder="Type pitcher name")

    count_input = st.text_input("Count (balls-strikes)", value="0-0", placeholder="e.g. 0-0, 1-2")

    st.markdown("---")
    st.markdown("### Game Situation")

    col_diamond, col_outs = st.columns([3, 2])

    with col_diamond:
        st.markdown("**Bases - Check boxes to add runners**")

        # Use checkboxes that directly update and persist
        base_col1, base_col2, base_col3 = st.columns(3)
        with base_col1:
            on_1b = st.checkbox("Runner on 1st", value=st.session_state.get("setup_on_1b", False), key="check_1b")
            st.session_state["setup_on_1b"] = on_1b
        with base_col2:
            on_2b = st.checkbox("Runner on 2nd", value=st.session_state.get("setup_on_2b", False), key="check_2b")
            st.session_state["setup_on_2b"] = on_2b
        with base_col3:
            on_3b = st.checkbox("Runner on 3rd", value=st.session_state.get("setup_on_3b", False), key="check_3b")
            st.session_state["setup_on_3b"] = on_3b

        # Then show the diamond visual
        diamond_svg = render_baseball_diamond(
            st.session_state['setup_on_1b'],
            st.session_state['setup_on_2b'],
            st.session_state['setup_on_3b'],
            st.session_state['setup_outs']
        )
        st.markdown(diamond_svg, unsafe_allow_html=True)

    with col_outs:
        st.markdown("**Outs**")
        outs = st.radio("", options=[0, 1, 2], index=st.session_state["setup_outs"],
                        format_func=lambda x: f"{x} out" + ("s" if x != 1 else ""),
                        key="outs_radio")
        st.session_state["setup_outs"] = outs

    # Store current selections for form submission
    on_1b = st.session_state["setup_on_1b"]
    on_2b = st.session_state["setup_on_2b"]
    on_3b = st.session_state["setup_on_3b"]

    st.markdown("---")
    submit = st.button("Start At-Bat", type="primary", use_container_width=True)

if submit:
    # Resolve selected names to ids (keep names-only in UI)
    try:
        if batter_names:
            batter_id = batter_name_to_id.get(batter_name)
        else:
            batter_id = batter_name if batter_name else None
    except Exception:
        batter_id = batter_name

    try:
        if pitcher_names:
            pitcher_id = pitcher_name_to_id.get(pitcher_name)
        else:
            pitcher_id = pitcher_name if pitcher_name else None
    except Exception:
        pitcher_id = pitcher_name

    if batter_id is None or pitcher_id is None:
        st.error("Could not resolve batter or pitcher from the provided names. Ensure mapping CSVs exist in data/raw and names are selected.")
        st.stop()

    # initialize session state for the stepwise at-bat
    st.session_state["atbat_active"] = True
    st.session_state.setdefault("history", [])
    st.session_state["batter_name"] = batter_name
    st.session_state["pitcher_name"] = pitcher_name
    st.session_state["batter_id"] = str(batter_id)
    st.session_state["pitcher_id"] = str(pitcher_id)
    st.session_state["count"] = count_input
    # set base occupancy and outs in situation ONLY if not already set (don't overwrite during at-bat)
    if "situation" not in st.session_state or len(st.session_state.get("history", [])) == 0:
        sit = {"on_1b": bool(on_1b), "on_2b": bool(on_2b), "on_3b": bool(on_3b), "outs": int(outs)}
        st.session_state["situation"] = sit


# Running at-bat UI: show whenever session indicates an active at-bat
if st.session_state.get("atbat_active"):
    # running at-bat UI
    batter_name = st.session_state.get("batter_name")
    pitcher_name = st.session_state.get("pitcher_name")
    batter_id = st.session_state.get("batter_id")
    pitcher_id = st.session_state.get("pitcher_id")
    count_input = st.session_state.get("count")
    situation = st.session_state.get("situation")

    # Matchup header
    archetype = get_archetype(arche_df, batter_id)
    arsenal_list = get_pitcher_arsenal(ars_df, pitcher_id)
    if arsenal_list:
        arsenal_display = ', '.join(arsenal_list)
    else:
        arsenal_display = "unknown"

    st.markdown(f"### {batter_name} vs {pitcher_name}")
    col_matchup1, col_matchup2 = st.columns(2)
    with col_matchup1:
        st.markdown(f"**Batter Archetype:** {archetype}")
    with col_matchup2:
        st.markdown(f"**Pitcher Arsenal:** {arsenal_display}")

    st.markdown("---")

    # Calculate count-specific run expectancy (will be used in compact display)
    try:
        balls_count, strikes_count = 0, 0
        if '-' in count_input:
            parts = count_input.split('-')
            balls_count = int(parts[0])
            strikes_count = int(parts[1])
    except:
        balls_count, strikes_count = 0, 0

    outs = int(situation.get("outs", 0))
    bases_state = f"{1 if situation.get('on_1b') else 0}{1 if situation.get('on_2b') else 0}{1 if situation.get('on_3b') else 0}"
    sit_rep = (outs, bases_state)
    base_re = _RE_MAP.get(sit_rep, 0.0)

    # Adjust RE based on count favorability
    count_adjustment = 0.0
    if strikes_count > balls_count:
        count_adjustment = -0.10 * base_re * (strikes_count - balls_count) / 2
    elif balls_count > strikes_count:
        count_adjustment = 0.10 * base_re * (balls_count - strikes_count) / 3

    if balls_count == 3 and strikes_count < 2:
        count_adjustment = 0.15 * base_re
    elif strikes_count == 2 and balls_count < 2:
        count_adjustment = -0.15 * base_re
    elif balls_count == 3 and strikes_count == 2:
        count_adjustment = 0.0

    current_re = base_re + count_adjustment

    # Compact game state display in a single row
    state_col1, state_col2, state_col3, state_col4 = st.columns([1, 1, 1, 1])

    with state_col1:
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #eff6ff; border-radius: 8px;'><div style='font-size: 12px; color: #6b7280; margin-bottom: 4px;'>COUNT</div><div style='font-size: 24px; font-weight: bold; color: #1e40af;'>{count_input}</div></div>", unsafe_allow_html=True)

    with state_col2:
        # Compact bases display
        on_1 = "●" if situation.get('on_1b', False) else "○"
        on_2 = "●" if situation.get('on_2b', False) else "○"
        on_3 = "●" if situation.get('on_3b', False) else "○"
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #f0fdf4; border-radius: 8px;'><div style='font-size: 12px; color: #6b7280; margin-bottom: 4px;'>BASES</div><div style='font-size: 20px;'>{on_3}<br>{on_2} {on_1}</div></div>", unsafe_allow_html=True)

    with state_col3:
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #fef3c7; border-radius: 8px;'><div style='font-size: 12px; color: #6b7280; margin-bottom: 4px;'>OUTS</div><div style='font-size: 24px; font-weight: bold; color: #92400e;'>{outs}</div></div>", unsafe_allow_html=True)

    with state_col4:
        st.markdown(f"<div style='text-align: center; padding: 10px; background: #f0fdf4; border: 2px solid #22c55e; border-radius: 8px;'><div style='font-size: 12px; color: #6b7280; margin-bottom: 4px;'>RUN EXP</div><div style='font-size: 24px; font-weight: bold; color: #166534;'>{current_re:.2f}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # At-bat history in collapsible section
    with st.expander("At-Bat History", expanded=False):
        history = st.session_state.get("history", [])
        render_pitch_history(history)

    # Show next-pitch candidates with percentages
    st.subheader("Next-pitch candidates")

    # If we have a trained model, use it to score the actual pitcher arsenal
    candidates = []
    if MODEL_PATH.exists():
        try:
            pid = int(pitcher_id) if str(pitcher_id).isdigit() else pitcher_id
            # build model_state from session
            count_val = st.session_state.get("count", "0-0")
            balls_val, strikes_val = 0, 0
            try:
                if '-' in count_val:
                    b_s = count_val.split('-')
                    balls_val = max(0, min(3, int(b_s[0])))
                    strikes_val = max(0, min(2, int(b_s[1])))
            except Exception:
                balls_val, strikes_val = 0, 0

            sit = st.session_state.get("situation", {})
            outs_val = int(sit.get("outs", 0))
            on_1b_val = 1 if sit.get("on_1b") else 0
            on_2b_val = 1 if sit.get("on_2b") else 0
            on_3b_val = 1 if sit.get("on_3b") else 0

            model_state = {
                "balls": int(balls_val),
                "strikes": int(strikes_val),
                "outs_when_up": int(outs_val),
                "on_1b": on_1b_val,
                "on_2b": on_2b_val,
                "on_3b": on_3b_val,
                "release_speed": 94.0,
                "release_spin_rate": 2200.0,
                "zone": 5,
                # if user selected a visual target, map it to plate coordinates
                "plate_x": 0.0,
                "plate_z": 2.5,
                "last_pitch_speed_delta": 0.0,
                "batter_pitchtype_woba": 0.0,
                "batter_pitchtype_whiff_rate": 0.0,
                "batter_pitchtype_run_value": 0.0,
                "stand": "R",
                "p_throws": "R",
                "pitch_type": None,
                "last_pitch_type": None,
                "last_pitch_result": None
            }

            # Use PitchRecommender for intelligent recommendations
            try:
                batter_id_val = st.session_state.get('batter_id')

                # Get last pitch type from history if available
                last_pitch_type = None
                if st.session_state.get("history"):
                    last_pitch_type = st.session_state["history"][-1].get("pitch")

                # Call the recommender
                if pitch_recommender is not None:
                    results = pitch_recommender.recommend(
                        pitcher_id=pid,
                        batter_id=batter_id_val,
                        balls=balls_val,
                        strikes=strikes_val,
                        outs=outs_val,
                        on_1b=bool(on_1b_val),
                        on_2b=bool(on_2b_val),
                        on_3b=bool(on_3b_val),
                        stand="R",  # TODO: Get from batter profile
                        last_pitch_type=last_pitch_type,
                        top_n=None  # Get all pitches in arsenal
                    )

                    # Store debug info for later display
                    debug_info = {
                        "model_used": True,
                        "pitcher_id": pid,
                        "batter_id": batter_id_val,
                        "num_recommendations": len(results)
                    }

                    # Convert results to candidates format for UI
                    for r in results:
                        pitch_type = r['pitch_type']
                        loc_data = r['location']

                        # Convert plate_x, plate_z to human-readable location string
                        px = loc_data.get('plate_x', 0.0)
                        pz = loc_data.get('plate_z', 2.5)

                        # Map to human-readable location
                        if px < -0.5:
                            horizontal = "inside"
                        elif px > 0.5:
                            horizontal = "outside"
                        else:
                            horizontal = "middle"

                        if pz < 2.3:
                            vertical = "low"
                        elif pz > 3.2:
                            vertical = "high"
                        else:
                            vertical = "mid"

                        loc_string = f"{horizontal}-{vertical}"

                        candidates.append({
                            "pitch": pitch_type,
                            "pct": round(r['probability_pct'], 1),
                            "location": loc_string,
                            "location_data": loc_data,  # Store raw location data for visualization
                            "expected_run_value": r['predicted_rv'],
                            "expected_after_re": r['expected_re_after']
                        })
                else:
                    raise Exception("PitchRecommender not available")
            except Exception as me:
                # model failed; fallback to heuristic
                candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)
        except Exception:
            candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)
    else:
        candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)

    # Add run expectancy values to heuristic candidates if not already present
    if candidates and not any('expected_after_re' in c for c in candidates):
        # Calculate current RE for this situation
        sit_rep = (int(situation.get("outs", 0)),
                   f"{1 if situation.get('on_1b') else 0}{1 if situation.get('on_2b') else 0}{1 if situation.get('on_3b') else 0}")
        current_re = _RE_MAP.get(sit_rep, 0.0)
        # Add expected_after_re to each candidate (heuristic doesn't predict actual run values, so display current RE)
        for c in candidates:
            c['expected_after_re'] = round(current_re, 2)

    # Determine full pitcher arsenal (all pitch types this pitcher throws)
    arsenal_list = get_pitcher_arsenal(ars_df, pitcher_id)

    if not arsenal_list:
        # try to resolve pitcher id from mapping file if available
        resolved_pid = pitcher_id
        try:
            # if pitcher_name exists in mapping, use its id
            mapped = pitcher_name_to_id.get(pitcher_name)
            if mapped:
                resolved_pid = mapped
                arsenal_list = get_pitcher_arsenal(ars_df, resolved_pid)
        except Exception:
            pass

        # if still empty, try substring match against arsenals values (maybe pitcher id recorded as different form)
        if not arsenal_list:
            def row_contains_name(r):
                return any(str(v).lower().find(str(pitcher_name).lower()) != -1 for v in r.values)
            try:
                matches = ars_df[ars_df.apply(row_contains_name, axis=1)]
                if not matches.empty:
                    # Get first match's arsenal
                    arsenal_list = get_pitcher_arsenal(ars_df, matches.iloc[0]["pitcher"])
            except Exception:
                pass

        # still empty: fallback to common pitches
        if not arsenal_list:
            arsenal_list = ["FF", "SL", "CU"]

    # Two-column layout: recommendations on left, pitch submission on right
    rec_col, form_col = st.columns([2, 1])

    with rec_col:
        st.markdown("#### Pitch Recommendations")
        if not candidates:
            st.info("No candidates available")
        else:
            # determine top recommended pitch (highest pct)
            try:
                top_pitch = max(candidates, key=lambda x: x.get('pct', 0))
                top_name = top_pitch.get('pitch')
            except Exception:
                top_pitch = None
                top_name = None

            # Show all recommendations in compact format
            for c in candidates:
                is_top = (c.get('pitch') == top_name)
                pitch_type = c.get('pitch')
                pitch_color = get_pitch_color(pitch_type)
                pct = c.get('pct', 0)
                expected_re = c.get('expected_after_re', 0)

                # Compact card styling
                border_style = f"border-left: 4px solid {pitch_color}; background: #f9fafb;" if is_top else f"border-left: 2px solid #e5e7eb; background: #ffffff;"

                # Get location data for strike zone
                loc_data = c.get('location_data', {})
                if loc_data and 'plate_x' in loc_data and 'plate_z' in loc_data:
                    cx, cy = _plate_xy_to_svg_coords(loc_data['plate_x'], loc_data['plate_z'])
                else:
                    loc = c.get('location', '')
                    cx, cy = _loc_to_svg_coords(loc)

                r = max(3, min(8, pct / 6))
                loc_string = c.get('location', '')

                if loc_data and 'plate_x' in loc_data:
                    tooltip = f"{pitch_type} {pct}% — Location: ({loc_data['plate_x']:.2f}, {loc_data['plate_z']:.2f}) — Expected RV: {loc_data.get('expected_rv', 0):.3f}"
                else:
                    tooltip = f"{pitch_type} {pct}% — Location: {loc_string} — Expected after RE: {expected_re}"

                strike_zone_svg = f'<svg width="100" height="120" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><title>{tooltip}</title><rect x="15" y="8" width="70" height="84" fill="#f0f9ff" stroke="#1e40af" stroke-width="2" rx="4" /><line x1="15" y1="36" x2="85" y2="36" stroke="#cbd5e1" stroke-width="0.5"/><line x1="15" y1="64" x2="85" y2="64" stroke="#cbd5e1" stroke-width="0.5"/><line x1="38" y1="8" x2="38" y2="92" stroke="#cbd5e1" stroke-width="0.5"/><line x1="62" y1="8" x2="62" y2="92" stroke="#cbd5e1" stroke-width="0.5"/><circle cx="{cx}" cy="{cy}" r="{r}" fill="{pitch_color}" opacity="0.85" stroke="#fff" stroke-width="2"/></svg>'

                card_html = f'''
                <div style='{border_style} border-radius: 8px; padding: 12px; margin-bottom: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);'>
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div style='flex: 1;'>
                            <div style='font-size: 20px; font-weight: bold; color: {pitch_color};'>
                                {pitch_type} {"★" if is_top else ""}
                            </div>
                            <div style='font-size: 16px; color: #374151;'>
                                <strong>{pct}%</strong> · RE: {expected_re:.2f}
                            </div>
                            <div style='font-size: 11px; color: #9ca3af;'>
                                {loc_string}
                            </div>
                        </div>
                        <div style='flex: 0 0 auto;'>
                            {strike_zone_svg}
                        </div>
                    </div>
                </div>
                '''
                st.markdown(card_html, unsafe_allow_html=True)

    with form_col:
        st.markdown("#### Submit Pitch")

        # If we have model-based candidate probabilities, reorder arsenal_list to put top candidates first
        if candidates and any('pct' in c for c in candidates):
            score_map = {c['pitch']: c.get('pct', 0) for c in candidates}
            arsenal_list = sorted(arsenal_list, key=lambda p: -score_map.get(p, 0))

        with st.form(key='select_next'):
            pitch_options = arsenal_list
            st.selectbox("Pitch Type", options=pitch_options, key='selected_pitch')
            st.selectbox("Outcome", ["Ball", "Called Strike", "Swinging Strike", "Foul", "In play - out", "In play - single", "In play - double", "In play - triple", "In play - home run", "Other (end)"], key='selected_outcome')
            submit_pitch = st.form_submit_button("Submit Pitch", use_container_width=True)

    # Show debug info at bottom if available
    if 'debug_info' in locals():
        with st.expander("🔧 Debug: Model Internals", expanded=False):
            st.write(debug_info)

    if submit_pitch:
        # read values from session_state
        selected_pitch = st.session_state.get('selected_pitch')
        outcome = st.session_state.get('selected_outcome')
        continue_ab = True  # Will be set to False for walk, strikeout, or in-play outcomes

        # append to history
        rec = {"pitch": selected_pitch, "outcome": outcome, "count_before": count_input}
        st.session_state.setdefault("history", [])
        st.session_state["history"].append(rec)

        # update count roughly based on outcome
        b, s = 0, 0
        try:
            if '-' in count_input:
                parts = count_input.split('-')
                b = int(parts[0])
                s = int(parts[1])
        except Exception:
            b, s = 0, 0

        if outcome == "Ball":
            b = min(4, b + 1)
        elif outcome in ("Called Strike", "Swinging Strike"):
            s = min(3, s + 1)
        elif outcome == "Foul":
            if s < 2:
                s = min(2, s + 1)
        elif outcome.startswith("In play"):
            # end at-bat for simplicity
            continue_ab = False

        # convert walk/strikeout detection
        walk_or_strikeout = False
        if b >= 4:
            continue_ab = False
            walk_or_strikeout = True
            st.success("Walk (4 balls) - Count reset to 0-0 for next batter")
        if s >= 3:
            continue_ab = False
            walk_or_strikeout = True
            st.success("Strikeout (3 strikes) - Count reset to 0-0 for next batter")

        # update count: reset to 0-0 for walk/strikeout/in-play, otherwise update normally
        if walk_or_strikeout or outcome.startswith("In play"):
            st.session_state["count"] = "0-0"
        else:
            st.session_state["count"] = f"{b}-{s}"

        st.session_state["atbat_active"] = bool(continue_ab)

        # update outs for strikeout
        if s >= 3:
            current_outs = st.session_state["situation"].get("outs", 0)
            st.session_state["situation"]["outs"] = min(2, current_outs + 1)

        # update outs/history if in-play outcome
        if outcome.startswith("In play"):
            # simple heuristic: if 'out' in outcome mark an out
            if "out" in outcome:
                st.session_state["situation"]["outs"] = min(2, st.session_state["situation"].get("outs", 0) + 1)
            else:
                # on-hit, potentially update base occupancy (simple approach)
                # single -> runner advance to at least 1B
                if "single" in outcome:
                    st.session_state["situation"]["on_1b"] = True
                if "double" in outcome:
                    st.session_state["situation"]["on_2b"] = True
                if "home run" in outcome:
                    st.session_state["situation"]["on_1b"] = False
                    st.session_state["situation"]["on_2b"] = False
                    st.session_state["situation"]["on_3b"] = False
        # Rerun the app to show updated state with the new pitch recorded
        st.rerun()
    # Offer JSON export of the at-bat history and metadata
    import json as _json
    atbat_summary = {
        "batter_name": st.session_state.get("batter_name"),
        "pitcher_name": st.session_state.get("pitcher_name"),
        "batter_id": st.session_state.get("batter_id"),
        "pitcher_id": st.session_state.get("pitcher_id"),
        "count": st.session_state.get("count"),
        "situation": st.session_state.get("situation"),
        "history": st.session_state.get("history", [])
    }
    st.download_button("Download at-bat summary (JSON)", data=_json.dumps(atbat_summary, indent=2), file_name="atbat_summary.json", mime="application/json")

    # Optionally show model-based next pitch if model exists
    if MODEL_PATH.exists():
        st.markdown("---")
        with st.expander("Model Analysis (Advanced)", expanded=False):
            st.markdown("**Model-based next-pitch estimate**")
            st.write("This uses the trained run-value model to score individual pitch choices.")
            try:
                pid = int(pitcher_id) if str(pitcher_id).isdigit() else pitcher_id
                # parse count safely from session
                count_input = st.session_state.get("count", "0-0")
                balls_val = 0
                strikes_val = 0
                try:
                    if '-' in count_input:
                        b_s = count_input.split('-')
                        balls_val = max(0, min(3, int(b_s[0])))
                        strikes_val = max(0, min(2, int(b_s[1])))
                except Exception:
                    balls_val = 0
                    strikes_val = 0
                # get outs/base occupancy from session situation
                sit = st.session_state.get("situation", {})
                outs_val = int(sit.get("outs", 0))
                on_1b_val = 1 if sit.get("on_1b") else 0
                on_2b_val = 1 if sit.get("on_2b") else 0
                on_3b_val = 1 if sit.get("on_3b") else 0
                # Build a minimal state for model call — fill with defaults where missing
                model_state = {
                    "balls": int(balls_val),
                    "strikes": int(strikes_val),
                    "outs_when_up": int(outs_val),
                    "on_1b": on_1b_val,
                    "on_2b": on_2b_val,
                    "on_3b": on_3b_val,
                    "release_speed": 94.0,
                    "release_spin_rate": 2200.0,
                    "zone": 5,
                    "plate_x": 0.0,
                    "plate_z": 2.5,
                    "last_pitch_speed_delta": 0.0,
                    "batter_pitchtype_woba": 0.0,
                    "batter_pitchtype_whiff_rate": 0.0,
                    "batter_pitchtype_run_value": 0.0,
                    "stand": "R",
                    "p_throws": "R",
                    "pitch_type": None,
                    "last_pitch_type": None,
                    "last_pitch_result": None
                }
                # build batter stats map for display
                batter_id_val = st.session_state.get('batter_id')
                batter_stats_map = {}
                pitcher_ars = get_pitcher_arsenal(ars_df, pid)
                if not pitcher_ars:
                    pitcher_ars = ["FF", "SL", "CU"]
                for pt in pitcher_ars:
                    w, wh, rv = get_batter_pitchtype_stats(_batter_pitch_stats_df, batter_id_val, pt)
                    batter_stats_map[pt] = (w, wh, rv)

                ptype, val, preds, _ = recommend_next_pitch(model_state, pid, batter_stats_map=batter_stats_map)
                # compute expected after-run-expectancy
                sit = st.session_state.get('situation', {})
                sit_rep = (int(sit.get("outs", 0)), f"{1 if sit.get('on_1b') else 0}{1 if sit.get('on_2b') else 0}{1 if sit.get('on_3b') else 0}")
                current_re = _RE_MAP.get(sit_rep, 0.0)
                pitches = list(preds.keys())
                vals = [preds[p] for p in pitches]
                after_re = [current_re + v for v in vals]
                st.success(f"Model recommends: **{ptype}**  (expected run value {val:+.3f}, expected after RE {current_re+val:+.3f})")
                df = pd.DataFrame({"pitch_type": pitches, "expected_run_value": vals, "expected_after_re": after_re})
                st.dataframe(df.sort_values("expected_after_re"))
            except Exception as e:
                st.warning(f"Model could not be used: {e}")