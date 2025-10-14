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

with st.form(key='start_atbat'):
    col1, col2 = st.columns([2, 1])
    with col1:
        if batter_names:
            batter_name = st.selectbox("Batter (name only)", options=batter_names, index=0)
        else:
            batter_name = st.text_input("Batter name (no id)", value="")

        if pitcher_names:
            pitcher_name = st.selectbox("Pitcher (name only)", options=pitcher_names, index=0)
        else:
            pitcher_name = st.text_input("Pitcher name (no id)", value="")

        count_input = st.text_input("Count (balls-strikes)", value="0-0")
    with col2:
        st.subheader("Situation")
        risp = st.checkbox("Runners in scoring position (RISP)")
        outs = st.selectbox("Outs", [0, 1, 2], index=0)
        late = st.checkbox("Late inning / high leverage")

    submit = st.form_submit_button("Start at-bat")

if submit:
    # Prepare situation
    situation = {"risp": bool(risp), "outs": int(outs), "late_inning": bool(late)}

    # Resolve selected names to ids (keep names-only in UI)
    # Batter
    try:
        if batter_names:
            batter_id = batter_name_to_id.get(batter_name)
        else:
            # no mapping available — try to use raw input as id or name
            batter_id = batter_name if batter_name else None
    except Exception:
        batter_id = batter_name

    # Pitcher
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
    st.session_state.setdefault("atbat_active", True)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("batter_name", batter_name)
    st.session_state.setdefault("pitcher_name", pitcher_name)
    st.session_state.setdefault("batter_id", str(batter_id))
    st.session_state.setdefault("pitcher_id", str(pitcher_id))
    st.session_state.setdefault("count", count_input)
    st.session_state.setdefault("situation", situation)

    st.experimental_rerun()
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
    candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, count_input, situation)

    # display as table with a mini strike-zone diagram per pitch (SVG)
    cols = st.columns([2, 3])
    with cols[0]:
        st.write("Pitch")
        for c in candidates:
            st.markdown(f"**{c['pitch']}** — {c['pct']}%")
    with cols[1]:
        st.write("Suggested locations (strike zone)")
        for c in candidates:
            # render a small SVG showing location text and a simple box
            loc = c.get("location", "")
            svg = f'''<svg width="160" height="160" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <rect x="20" y="10" width="60" height="80" fill="#ffffff" stroke="#000" />
                <text x="50" y="95" font-size="6" text-anchor="middle">{c['pitch']} {c['pct']}%</text>
                <circle cx="{50}" cy="{loc.startswith('low') and 70 or 30}" r="6" fill="#e53e3e" opacity="0.8" />
                <text x="50" y="50" font-size="5" text-anchor="middle">{loc}</text>
            </svg>'''
            st.write(svg, unsafe_allow_html=True)

    st.write("")
    st.markdown("Select which pitch will be thrown next and enter the pitch outcome.")

    with st.form(key='select_next'):
        pitch_options = [c['pitch'] for c in candidates]
        selected_pitch = st.selectbox("Select pitch", options=pitch_options)
        outcome = st.selectbox("Outcome", ["Ball", "Called Strike", "Swinging Strike", "Foul", "In play - out", "In play - single", "In play - double", "In play - triple", "In play - home run", "Other (end)"])
        continue_ab = st.checkbox("At-bat continues after this pitch", value=True)
        submit_pitch = st.form_submit_button("Submit pitch")

    if submit_pitch:
        # append to history
        rec = {"pitch": selected_pitch, "outcome": outcome, "count_before": count_input}
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

        # update count
        st.session_state["count"] = f"{b}-{s}"
        st.session_state["atbat_active"] = bool(continue_ab)

        # rerun to refresh UI
        st.experimental_rerun()

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
            # get outs/risp from session situation
            sit = st.session_state.get("situation", {})
            outs_val = int(sit.get("outs", 0))
            risp_val = bool(sit.get("risp", False))
            # Build a minimal state for model call — fill with defaults where missing
            model_state = {
                "balls": int(balls_val),
                "strikes": int(strikes_val),
                "outs_when_up": int(outs_val),
                "on_1b": 0,
                "on_2b": 1 if risp_val else 0,
                "on_3b": 0,
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
            ptype, val, preds = recommend_next_pitch(model_state, pid)
            st.success(f"Model recommends: **{ptype}**  (expected run value {val:+.3f})")
            df = pd.DataFrame({"pitch_type": list(preds.keys()), "expected_run_value": list(preds.values())})
            st.dataframe(df.sort_values("expected_run_value"))
        except Exception as e:
            st.warning(f"Model could not be used: {e}")


