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
    from attack_recommender import load_data, recommend_sequence, get_archetype, get_next_pitch_candidates, map_pitch_to_location, load_batter_pitchtype_stats, get_batter_pitchtype_stats
    from predict import recommend_next_pitch, MODEL_PATH
except Exception as e:
    # Don't raise here — show a helpful message in the UI so deployment isn't a blank page.
    IMPORT_ERROR = e

st.set_page_config(page_title="PitchSequence", layout="centered")
st.title("PitchSequence — Pitch Sequence Recommender")

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
    import traceback
    st.error("Failed to import project modules required by the app. See details below.")
    st.text(traceback.format_exc())
    st.stop()


# Safe experimental rerun helper: prevents AttributeError when deployed
# to environments with older/newer Streamlit versions that don't expose
# the experimental API. It's intentionally a no-op when unavailable.
def _safe_rerun():
    try:
        if hasattr(st, "experimental_rerun") and callable(st.experimental_rerun):
            st.experimental_rerun()
    except Exception:
        return

# Load data
ARCH_PATH = Path("data/player_archetypes.csv")
ARS_PATH = Path("pitcher_assets/pitcher_arsenals.csv")
arche_df, ars_df = load_data(ARCH_PATH, ARS_PATH)

# load batter vs pitch_type matchup stats (if processed features exist)
_batter_pitch_stats_df = load_batter_pitchtype_stats()


def _load_run_expectancy(path: str = "static/run_expectancy_24.csv"):
    p = Path(path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        return {(int(r.outs), str(r.bases_state)): float(r.run_expectancy) for r in df.itertuples()}
    except Exception:
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

st.markdown("Select a batter and pitcher, set the count and base occupancy, then click 'Start at-bat'. Visuals show suggested locations and model probabilities.")
st.session_state.setdefault("atbat_active", False)

# ensure submit exists even if form isn't rendered
submit = False

if not st.session_state.get("atbat_active"):
    with st.form(key='start_atbat'):
        col1, col2 = st.columns([2, 1])
        with col1:
            # Batter selectbox (names-only)
            if batter_names:
                batter_name = st.selectbox("Batter (select name)", options=batter_names, index=0, help="Choose batter by name; maps to internal id automatically")
            else:
                batter_name = st.text_input("Batter name (no id)", value="", placeholder="Type batter name")

            # Pitcher selectbox (names-only)
            if pitcher_names:
                pitcher_name = st.selectbox("Pitcher (select name)", options=pitcher_names, index=0, help="Choose pitcher by name; known arsenals will be used")
            else:
                pitcher_name = st.text_input("Pitcher name (no id)", value="", placeholder="Type pitcher name")

            count_input = st.text_input("Count (balls-strikes)", value="0-0", placeholder="e.g. 0-0, 1-2")

        with col2:
            st.subheader("Situation — base occupancy")
            st.markdown("Select which bases are occupied (visual triangle).")
            b1, b2, b3 = st.columns(3)
            with b1:
                on_1b = st.checkbox("1B")
            with b2:
                on_2b = st.checkbox("2B")
            with b3:
                on_3b = st.checkbox("3B")

            outs = st.selectbox("Outs", [0, 1, 2], index=0)

        submit = st.form_submit_button("Start at-bat")

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
    # set base occupancy and outs in situation
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

    archetype = get_archetype(arche_df, batter_id)
    st.markdown(f"**Batter:** {batter_name} — **Archetype:** {archetype}")
    # show pitcher arsenal if available (use resolved pitcher_id)
    pitcher_row = ars_df[ars_df["pitcher"].astype(str) == str(pitcher_id)]
    if not pitcher_row.empty:
        arsenal_list = pitcher_row["pitch_type"].astype(str).unique().tolist()
        st.markdown(f"**Pitcher:** {pitcher_name} — arsenal: {', '.join(arsenal_list)}")
    else:
        st.markdown(f"**Pitcher:** {pitcher_name} — arsenal: unknown")

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

            # model returns (best_pitch, best_val, preds_dict)
            try:
                # build batter stats map for this batter (pitch -> (woba, whiff, run_value))
                batter_id_val = st.session_state.get('batter_id')
                batter_stats_map = {}
                # get the pitcher's arsenal and construct stats for each pitch (fallback zeros when missing)
                try:
                    pitcher_ars = ars_df[ars_df["pitcher"].astype(str) == str(pid)]["pitch_type"].astype(str).unique().tolist()
                except Exception:
                    pitcher_ars = []
                if not pitcher_ars:
                    # fallback to some common pitch types
                    pitcher_ars = ["FF", "SL", "CU"]

                for pt in pitcher_ars:
                    w, wh, rv = get_batter_pitchtype_stats(_batter_pitch_stats_df, batter_id_val, pt)
                    batter_stats_map[pt] = (w, wh, rv)

                best_pt, best_val, preds = recommend_next_pitch(model_state, pid, batter_stats_map=batter_stats_map)
                # expose debug info for verification
                with st.expander("Debug: model internals", expanded=False):
                    st.write({"model_used": True, "pitcher_id": pid, "batter_id": batter_id_val})
                    st.write("batter_stats_map sample:", {k: batter_stats_map[k] for k in list(batter_stats_map)[:10]})
                    st.write("raw_preds:", preds)
                # preds: mapping pitch -> expected_run_value (lower is better).
                # Convert to probabilities using run expectancy: compute current RE and expected_after_RE = current_RE + rv
                # then p ~ softmax(-expected_after_RE)
                import math
                pitches = list(preds.keys())
                vals = [preds[p] for p in pitches]
                # determine current run expectancy from situation
                sit_rep = (int(situation.get("outs", 0)), f"{1 if situation.get('on_1b') else 0}{1 if situation.get('on_2b') else 0}{1 if situation.get('on_3b') else 0}")
                current_re = _RE_MAP.get(sit_rep, 0.0)
                after_re = [current_re + v for v in vals]
                neg = [-v for v in after_re]
                max_neg = max(neg) if neg else 0.0
                exps = [math.exp(x - max_neg) for x in neg]
                s = sum(exps) if exps else 1.0
                probs = [e / s for e in exps]

                for p, prob, rv in zip(pitches, probs, vals):
                    loc = map_pitch_to_location(p, get_archetype(arche_df, batter_id), situation)
                    candidates.append({"pitch": p, "pct": round(prob * 100, 1), "location": loc, "expected_run_value": round(rv, 4), "expected_after_re": round(current_re + rv, 4)})
            except Exception as me:
                # model failed; fallback to heuristic
                candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)
        except Exception:
            candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)
    else:
        candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)

    # Determine full pitcher arsenal (all pitch types this pitcher throws)
    pitcher_row = ars_df[ars_df["pitcher"].astype(str) == str(pitcher_id)]
    if not pitcher_row.empty:
        arsenal_list = pitcher_row["pitch_type"].astype(str).unique().tolist()
    else:
        # try to resolve pitcher id from mapping file if available
        # and re-check arsenals
        # mapping loaded earlier: pitcher_name_to_id
        resolved_pid = pitcher_id
        try:
            # if pitcher_name exists in mapping, use its id
            mapped = pitcher_name_to_id.get(pitcher_name)
            if mapped:
                resolved_pid = mapped
                pitcher_row = ars_df[ars_df["pitcher"].astype(str) == str(resolved_pid)]
        except Exception:
            pass

        # if still empty, try substring match against arsenals values (maybe pitcher id recorded as different form)
        if pitcher_row.empty:
            def row_contains_name(r):
                return any(str(v).lower().find(str(pitcher_name).lower()) != -1 for v in r.values)
            matches = ars_df[ars_df.apply(row_contains_name, axis=1)]
            if not matches.empty:
                arsenal_list = matches["pitch_type"].astype(str).unique().tolist()
        else:
            arsenal_list = pitcher_row["pitch_type"].astype(str).unique().tolist()

        # still empty: fallback to the most common arsenal items across dataset
        if not ( 'arsenal_list' in locals() and arsenal_list ):
            common = ars_df["pitch_type"].value_counts().index.tolist()
            # take top 3 most common pitch types league-wide
            arsenal_list = common[:3] if common else ["FF", "SL", "CU"]

    # display candidates as paired rows: text info on left, strike-zone visual on right
    st.markdown("**Next-pitch candidates (click Submit to record a pitch)**")
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

        for c in candidates:
            is_top = (c.get('pitch') == top_name)
            # row container
            with st.container():
                left, right = st.columns([1, 1])
                with left:
                    # highlight top candidate
                    if is_top:
                        st.markdown(f"<div style='border-left:4px solid #2f855a; padding-left:8px'><strong>{c['pitch']}</strong> — {c.get('pct','?')}%<br/><small>expected after RE: {c.get('expected_after_re', '?')}</small></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='padding-left:8px'><strong>{c['pitch']}</strong> — {c.get('pct','?')}%<br/><small>expected after RE: {c.get('expected_after_re', '?')}</small></div>", unsafe_allow_html=True)
                with right:
                    loc = c.get('location', '')
                    cx, cy = _loc_to_svg_coords(loc)
                    r, color = _prob_to_visuals(c.get('pct', 0))
                    # slightly larger SVG for clarity; include title for tooltip
                    svg = f'''<svg width="200" height="220" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <title>{c.get('pitch')} {c.get('pct','?')}% — expected after RE: {c.get('expected_after_re','?')}</title>
                        <rect x="15" y="8" width="70" height="84" fill="#ffffff" stroke="#000" rx="4" />
                        <text x="50" y="20" font-size="4" text-anchor="middle">{loc}</text>
                        <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.95" />
                        <text x="50" y="98" font-size="5" text-anchor="middle">{c.get('pitch')} {c.get('pct','?')}%</text>
                    </svg>'''
                    # add subtle border if top
                    if is_top:
                        st.markdown(f"<div style='border:2px solid #2f855a; display:inline-block; padding:6px'>{svg}</div>", unsafe_allow_html=True)
                    else:
                        st.write(svg, unsafe_allow_html=True)

                    # (visual target feature removed for clarity)

    st.write("")
    st.markdown("Select which pitch will be thrown next and enter the pitch outcome.")

    # If we have model-based candidate probabilities, reorder arsenal_list to put top candidates first
    if candidates and any('pct' in c for c in candidates):
        # build a mapping pitch -> pct
        score_map = {c['pitch']: c.get('pct', 0) for c in candidates}
        # sort arsenal_list by pct desc, fallback to original order
        arsenal_list = sorted(arsenal_list, key=lambda p: -score_map.get(p, 0))

    with st.form(key='select_next'):
        # show full pitcher arsenal as the selectable options
        pitch_options = arsenal_list
        # use explicit keys so values persist in session state across reruns
        st.selectbox("Select pitch", options=pitch_options, key='selected_pitch')
        st.selectbox("Outcome", ["Ball", "Called Strike", "Swinging Strike", "Foul", "In play - out", "In play - single", "In play - double", "In play - triple", "In play - home run", "Other (end)"], key='selected_outcome')
        st.checkbox("At-bat continues after this pitch", value=True, key='selected_continue_ab')
        submit_pitch = st.form_submit_button("Submit pitch")

    if submit_pitch:
        # read values from session_state
        selected_pitch = st.session_state.get('selected_pitch')
        outcome = st.session_state.get('selected_outcome')
        continue_ab = st.session_state.get('selected_continue_ab', True)

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
        if b >= 4:
            continue_ab = False
            st.success("Walk (4 balls)")
        if s >= 3:
            continue_ab = False
            st.success("Strikeout (3 strikes)")

        # update count and at-bat state using the local continue_ab
        st.session_state["count"] = f"{b}-{s}"
        st.session_state["atbat_active"] = bool(continue_ab)

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
        # show a small confirmation so user sees the recorded pitch
        st.info(f"Recorded: {selected_pitch} -> {outcome}. Count now {st.session_state['count']}")

        # recompute candidates using the updated session state so the UI reflects new probabilities
        # (duplicate of the earlier candidate-generation logic but based on latest st.session_state)
        candidates = []
        try:
            if MODEL_PATH.exists():
                try:
                    pid = int(st.session_state.get('pitcher_id')) if str(st.session_state.get('pitcher_id')).isdigit() else st.session_state.get('pitcher_id')
                    # build model state from updated session
                    count_val = st.session_state.get('count', '0-0')
                    balls_val, strikes_val = 0, 0
                    try:
                        if '-' in count_val:
                            b_s = count_val.split('-')
                            balls_val = max(0, min(3, int(b_s[0])))
                            strikes_val = max(0, min(2, int(b_s[1])))
                    except Exception:
                        balls_val, strikes_val = 0, 0

                    sit = st.session_state.get('situation', {})
                    outs_val = int(sit.get('outs', 0))
                    on_1b_val = 1 if sit.get('on_1b') else 0
                    on_2b_val = 1 if sit.get('on_2b') else 0
                    on_3b_val = 1 if sit.get('on_3b') else 0

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

                    # build batter stats map for updated session covering the pitcher's arsenal
                    batter_id_val = st.session_state.get('batter_id')
                    batter_stats_map = {}
                    try:
                        pitcher_ars = ars_df[ars_df["pitcher"].astype(str) == str(pid)]["pitch_type"].astype(str).unique().tolist()
                    except Exception:
                        pitcher_ars = ["FF", "SL", "CU"]
                    for pt in pitcher_ars:
                        w, wh, rv = get_batter_pitchtype_stats(_batter_pitch_stats_df, batter_id_val, pt)
                        batter_stats_map[pt] = (w, wh, rv)

                    best_pt, best_val, preds = recommend_next_pitch(model_state, pid, batter_stats_map=batter_stats_map)
                    with st.expander("Debug: updated model internals", expanded=False):
                        st.write({"model_used": True, "pitcher_id": pid, "batter_id": batter_id_val})
                        st.write("batter_stats_map sample:", {k: batter_stats_map[k] for k in list(batter_stats_map)[:10]})
                        st.write("raw_preds:", preds)
                    import math
                    pitches = list(preds.keys())
                    vals = [preds[p] for p in pitches]
                    # compute current RE based on updated situation
                    sit = st.session_state.get('situation', {})
                    sit_rep = (int(sit.get("outs", 0)), f"{1 if sit.get('on_1b') else 0}{1 if sit.get('on_2b') else 0}{1 if sit.get('on_3b') else 0}")
                    current_re = _RE_MAP.get(sit_rep, 0.0)
                    after_re = [current_re + v for v in vals]
                    neg = [-v for v in after_re]
                    max_neg = max(neg) if neg else 0.0
                    exps = [math.exp(x - max_neg) for x in neg]
                    s = sum(exps) if exps else 1.0
                    probs = [e / s for e in exps]

                    for p, prob, rv, aer in zip(pitches, probs, vals, after_re):
                        loc = map_pitch_to_location(p, get_archetype(arche_df, st.session_state.get('batter_id')), st.session_state.get('situation', {}))
                        candidates.append({"pitch": p, "pct": round(prob * 100, 1), "location": loc, "expected_run_value": round(rv, 4), "expected_after_re": round(aer, 4)})
                except Exception:
                    candidates = get_next_pitch_candidates(arche_df, ars_df, st.session_state.get('batter_id'), st.session_state.get('pitcher_id'), st.session_state.get('count'), st.session_state.get('situation', {}))
            else:
                candidates = get_next_pitch_candidates(arche_df, ars_df, st.session_state.get('batter_id'), st.session_state.get('pitcher_id'), st.session_state.get('count'), st.session_state.get('situation', {}))
        except Exception:
            candidates = get_next_pitch_candidates(arche_df, ars_df, st.session_state.get('batter_id'), st.session_state.get('pitcher_id'), st.session_state.get('count'), st.session_state.get('situation', {}))

        # render updated candidates below the confirmation so user sees new percentages immediately
        if not candidates:
            st.info("No updated candidates available")
        else:
            try:
                top_pitch = max(candidates, key=lambda x: x.get('pct', 0))
                top_name = top_pitch.get('pitch')
            except Exception:
                top_name = None
            for c in candidates:
                is_top = (c.get('pitch') == top_name)
                with st.container():
                    left, right = st.columns([1, 1])
                    with left:
                        if is_top:
                            st.markdown(f"<div style='border-left:4px solid #2f855a; padding-left:8px'><strong>{c['pitch']}</strong> — {c.get('pct','?')}%<br/><small>expected after RE: {c.get('expected_after_re', '?')}</small></div>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<div style='padding-left:8px'><strong>{c['pitch']}</strong> — {c.get('pct','?')}%<br/><small>expected after RE: {c.get('expected_after_re', '?')}</small></div>", unsafe_allow_html=True)
                    with right:
                        loc = c.get('location', '')
                        cx, cy = _loc_to_svg_coords(loc)
                        svg = f'''<svg width="200" height="220" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                            <rect x="15" y="8" width="70" height="84" fill="#ffffff" stroke="#000" rx="4" />
                            <text x="50" y="20" font-size="4" text-anchor="middle">{loc}</text>
                            <circle cx="{cx}" cy="{cy}" r="6" fill="#e53e3e" opacity="0.95" />
                            <text x="50" y="98" font-size="5" text-anchor="middle">{c.get('pitch')} {c.get('pct','?')}%</text>
                        </svg>'''
                        if is_top:
                            st.markdown(f"<div style='border:2px solid #2f855a; display:inline-block; padding:6px'>{svg}</div>", unsafe_allow_html=True)
                        else:
                            st.write(svg, unsafe_allow_html=True)
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
        st.subheader("Model-based next-pitch estimate")
        st.write("This uses the trained run-value model (if present) to score individual pitch choices.")
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
            try:
                pitcher_ars = ars_df[ars_df["pitcher"].astype(str) == str(pid)]["pitch_type"].astype(str).unique().tolist()
            except Exception:
                pitcher_ars = ["FF", "SL", "CU"]
            for pt in pitcher_ars:
                w, wh, rv = get_batter_pitchtype_stats(_batter_pitch_stats_df, batter_id_val, pt)
                batter_stats_map[pt] = (w, wh, rv)

            ptype, val, preds = recommend_next_pitch(model_state, pid, batter_stats_map=batter_stats_map)
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


