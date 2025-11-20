import streamlit as st
import pandas as pd
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

IMPORT_ERROR = None
try:
    from attack_recommender import load_data, recommend_sequence, get_archetype, get_next_pitch_candidates, map_pitch_to_location, load_batter_pitchtype_stats, get_batter_pitchtype_stats
    from predict import recommend_next_pitch, MODEL_PATH
except Exception as e:
    IMPORT_ERROR = e

st.set_page_config(page_title="PitchSequence", layout="centered")
st.title("PitchSequence — Pitch Sequence Recommender")

try:
    import subprocess
    short_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    branch_name = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    st.sidebar.markdown(f"**commit:** `{short_hash}`  \n**branch:** {branch_name}")
    st.info(f"Running commit: {short_hash} (branch: {branch_name})")
except Exception:
    pass

if IMPORT_ERROR is not None:
    import traceback
    st.error("Failed to import project modules required by the app. See details below.")
    st.text(traceback.format_exc())
    st.stop()

ARCH_PATH = Path("data/player_archetypes.csv")
ARS_PATH = Path("pitcher_assets/pitcher_arsenals.csv")
try:
    arche_df, ars_df = load_data(ARCH_PATH, ARS_PATH)
except FileNotFoundError as fe:
    # Handle missing files gracefully in deployed environments where data may live elsewhere
    st.sidebar.error(f"Data files not found: {fe}")
    arche_df = pd.DataFrame(columns=["batter", "cluster", "label"])
    ars_df = pd.DataFrame(columns=["pitcher", "pitch_type"])
except Exception as e:
    # Generic fallback: don't crash the app during startup; surface the error and continue with empty frames
    st.sidebar.error(f"Failed to load archetypes/arsenals: {e}")
    arche_df = pd.DataFrame(columns=["batter", "cluster", "label"])
    ars_df = pd.DataFrame(columns=["pitcher", "pitch_type"])

# Debug: Check if arsenals loaded correctly
if ars_df.empty or "pitcher" not in ars_df.columns:
    st.sidebar.warning("Arsenal data may have loading issues")
    st.sidebar.write(f"Arsenal columns: {list(ars_df.columns)}")
    st.sidebar.write(f"Arsenal shape: {ars_df.shape}")
    
    # Try to fix if header was skipped
    if ars_df.shape[1] == 2 and "pitcher" not in ars_df.columns:
        ars_df.columns = ["pitcher", "pitch_type"]
        st.sidebar.success("Fixed arsenal column names")

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
    x = 50
    y = 50
    if not loc:
        return x, y
    s = str(loc).lower()
    if 'in' in s:
        x = 65
    elif 'out' in s or 'away' in s:
        x = 35
    elif 'edge' in s:
        x = 30
    elif 'middle' in s:
        x = 50
    if 'low' in s:
        y = 70
    elif 'up' in s or 'high' in s:
        y = 30
    elif 'middle' in s or 'middle-strike' in s:
        y = 50
    if 'arm' in s:
        x = 60
    if 'edge-low-away' in s:
        x, y = 28, 72
    return x, y

def _prob_to_visuals(pct: float):
    try:
        p = float(pct)
    except Exception:
        p = 0.0
    r = 5 + (p / 100.0) * 7
    green = int(max(80, 200 - (p / 100.0) * 160))
    color = f"#e53e3e" if p < 1 else f"rgb(229,{green},62)"
    return r, color

BATTER_MAP_PATH = Path("data/raw/unique_batters_with_names.csv")
PITCHER_MAP_PATH = Path("data/raw/unique_pitchers_with_names.csv")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/christopherhsuu/pitchsequence/main/"

_MAP_SOURCES = {}

def _load_name_map(p: Path):
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
            continue
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
    _MAP_SOURCES[name] = None
    return {}, []

batter_name_to_id, batter_names = _load_name_map(BATTER_MAP_PATH)
pitcher_name_to_id, pitcher_names = _load_name_map(PITCHER_MAP_PATH)

st.markdown("Select a batter and pitcher, set the count and base occupancy, then click 'Start at-bat'. The system recommends pitches that minimize expected runs.")
st.session_state.setdefault("atbat_active", False)

submit = False

if not st.session_state.get("atbat_active"):
    with st.form(key='start_atbat'):
        col1, col2 = st.columns([2, 1])
        with col1:
            if batter_names:
                batter_name = st.selectbox("Batter (select name)", options=batter_names, index=0)
            else:
                batter_name = st.text_input("Batter name", value="", placeholder="Type batter name")

            if pitcher_names:
                pitcher_name = st.selectbox("Pitcher (select name)", options=pitcher_names, index=0)
            else:
                pitcher_name = st.text_input("Pitcher name", value="", placeholder="Type pitcher name")

            count_input = st.text_input("Count (balls-strikes)", value="0-0", placeholder="e.g. 0-0, 1-2")

        with col2:
            st.subheader("Situation — base occupancy")
            st.markdown("Select which bases are occupied")
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
    try:
        batter_id = batter_name_to_id.get(batter_name) if batter_names else batter_name
        pitcher_id = pitcher_name_to_id.get(pitcher_name) if pitcher_names else pitcher_name
    except Exception:
        batter_id = batter_name
        pitcher_id = pitcher_name

    if batter_id is None or pitcher_id is None:
        st.error("Could not resolve batter or pitcher. Ensure mapping CSVs exist and names are selected.")
        st.stop()

    st.session_state["atbat_active"] = True
    st.session_state.setdefault("history", [])
    st.session_state["batter_name"] = batter_name
    st.session_state["pitcher_name"] = pitcher_name
    st.session_state["batter_id"] = str(batter_id)
    st.session_state["pitcher_id"] = str(pitcher_id)
    st.session_state["count"] = count_input
    sit = {"on_1b": bool(on_1b), "on_2b": bool(on_2b), "on_3b": bool(on_3b), "outs": int(outs)}
    st.session_state["situation"] = sit


def generate_candidates(batter_id, pitcher_id, situation):
    """Generate pitch candidates ranked by expected run expectancy (lower is better for pitcher)."""
    candidates = []
    
    if not MODEL_PATH.exists():
        return get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, 
                                        st.session_state.get("count", "0-0"), situation)
    
    try:
        pid = int(pitcher_id) if str(pitcher_id).isdigit() else pitcher_id
        count_val = st.session_state.get("count", "0-0")
        balls_val, strikes_val = 0, 0
        
        if '-' in count_val:
            try:
                b_s = count_val.split('-')
                balls_val = max(0, min(3, int(b_s[0])))
                strikes_val = max(0, min(2, int(b_s[1])))
            except Exception:
                pass

        sit = st.session_state.get("situation", {})
        model_state = {
            "balls": int(balls_val),
            "strikes": int(strikes_val),
            "outs_when_up": int(sit.get("outs", 0)),
            "on_1b": 1 if sit.get("on_1b") else 0,
            "on_2b": 1 if sit.get("on_2b") else 0,
            "on_3b": 1 if sit.get("on_3b") else 0,
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

        batter_stats_map = {}
        # Check if columns exist in arsenal dataframe
        if "pitcher" in ars_df.columns and "pitch_type" in ars_df.columns:
            pitcher_ars = ars_df[ars_df["pitcher"].astype(str) == str(pid)]["pitch_type"].astype(str).unique().tolist()
        else:
            # If columns are missing, use first two columns as pitcher, pitch_type
            if ars_df.shape[1] >= 2:
                col0, col1 = ars_df.columns[0], ars_df.columns[1]
                pitcher_ars = ars_df[ars_df[col0].astype(str) == str(pid)][col1].astype(str).unique().tolist()
            else:
                pitcher_ars = []
        
        if not pitcher_ars:
            pitcher_ars = ["FF", "SL", "CU"]

        for pt in pitcher_ars:
            w, wh, rv = get_batter_pitchtype_stats(_batter_pitch_stats_df, batter_id, pt)
            batter_stats_map[pt] = (w, wh, rv)

        # recommend_next_pitch returns (best_pitch, best_value, predictions_dict)
        result = recommend_next_pitch(model_state, pid, batter_stats_map=batter_stats_map)
        
        # Handle the return value - it should be a tuple of (pitch, value, dict)
        if isinstance(result, tuple) and len(result) == 3:
            best_pt, best_val, preds = result
        else:
            # Fallback if return format is unexpected
            st.warning(f"Unexpected model return format: {type(result)}")
            raise ValueError("Model returned unexpected format")
        
        sit_rep = (int(situation.get("outs", 0)), 
                  f"{1 if situation.get('on_1b') else 0}{1 if situation.get('on_2b') else 0}{1 if situation.get('on_3b') else 0}")
        current_re = _RE_MAP.get(sit_rep, 0.0)
        
        pitch_data = []
        for pitch, run_val in preds.items():
            expected_after = current_re + run_val
            pitch_data.append({
                "pitch": pitch,
                "expected_run_value": run_val,
                "expected_after_re": expected_after
            })
        
        pitch_data.sort(key=lambda x: x["expected_after_re"])
        
        min_re = pitch_data[0]["expected_after_re"]
        max_re = pitch_data[-1]["expected_after_re"]
        range_re = max_re - min_re if max_re > min_re else 1.0
        
        for p_data in pitch_data:
            if range_re > 0:
                normalized = (p_data["expected_after_re"] - min_re) / range_re
                p_data["score"] = 1.0 - normalized
            else:
                p_data["score"] = 1.0
        
        total_score = sum(p["score"] for p in pitch_data)
        for p_data in pitch_data:
            p_data["pct"] = round((p_data["score"] / total_score) * 100, 1) if total_score > 0 else 0
            loc = map_pitch_to_location(p_data["pitch"], get_archetype(arche_df, batter_id), situation)
            p_data["location"] = loc
        
        candidates = pitch_data
        
        with st.expander("Model Debug Info", expanded=False):
            st.write(f"**Current RE:** {current_re:.3f}")
            try:
                best_val_float = float(best_val)
                st.write(f"**Best Pitch:** {best_pt} (Expected after RE: {current_re + best_val_float:.3f})")
            except (ValueError, TypeError):
                st.write(f"**Best Pitch:** {best_pt} (Run value: {best_val})")
            st.dataframe(pd.DataFrame(candidates))
            
    except Exception as e:
        st.error(f"Model error: {e}")
        import traceback
        st.text(traceback.format_exc())
        candidates = get_next_pitch_candidates(arche_df, ars_df, batter_id, pitcher_id, 
                                              st.session_state.get("count", "0-0"), situation)
    
    return candidates


if st.session_state.get("atbat_active"):
    batter_name = st.session_state.get("batter_name")
    pitcher_name = st.session_state.get("pitcher_name")
    batter_id = st.session_state.get("batter_id")
    pitcher_id = st.session_state.get("pitcher_id")
    count_input = st.session_state.get("count")
    situation = st.session_state.get("situation")

    archetype = get_archetype(arche_df, batter_id)
    st.markdown(f"**Batter:** {batter_name} — **Archetype:** {archetype}")
    
    # Check if columns exist in arsenal dataframe
    if "pitcher" in ars_df.columns and "pitch_type" in ars_df.columns:
        pitcher_row = ars_df[ars_df["pitcher"].astype(str) == str(pitcher_id)]
    else:
        # Use first two columns if standard columns missing
        if ars_df.shape[1] >= 2:
            col0 = ars_df.columns[0]
            pitcher_row = ars_df[ars_df[col0].astype(str) == str(pitcher_id)]
        else:
            pitcher_row = pd.DataFrame()
    
    if not pitcher_row.empty:
        if "pitch_type" in ars_df.columns:
            arsenal_list = pitcher_row["pitch_type"].astype(str).unique().tolist()
        else:
            arsenal_list = pitcher_row[ars_df.columns[1]].astype(str).unique().tolist()
        st.markdown(f"**Pitcher:** {pitcher_name} — arsenal: {', '.join(arsenal_list)}")
    else:
        arsenal_list = ["FF", "SL", "CU"]
        st.markdown(f"**Pitcher:** {pitcher_name} — arsenal: unknown (using defaults)")

    st.subheader("Pitch Recommendations")
    st.caption("Ranked by expected run expectancy after pitch (lower is better for pitcher)")

    candidates = generate_candidates(batter_id, pitcher_id, situation)

    if not candidates:
        st.info("No candidates available")
    else:
        sit_rep = (int(situation.get("outs", 0)), 
                  f"{1 if situation.get('on_1b') else 0}{1 if situation.get('on_2b') else 0}{1 if situation.get('on_3b') else 0}")
        current_re = _RE_MAP.get(sit_rep, 0.0)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Count", count_input)
        with col2:
            st.metric("Current RE", f"{current_re:.3f}")
        with col3:
            st.metric("Outs", situation.get("outs", 0))
        
        st.markdown("---")
        
        top_name = candidates[0].get('pitch')
        
        for c in candidates:
            is_top = (c.get('pitch') == top_name)
            
            with st.container():
                left, right = st.columns([1, 1])
                
                with left:
                    if is_top:
                        # Convert to float safely
                        try:
                            after_re_val = float(c.get('expected_after_re', 0))
                            run_val = float(c.get('expected_run_value', 0))
                        except (ValueError, TypeError):
                            after_re_val = 0.0
                            run_val = 0.0
                        
                        st.markdown(f"""
                        <div style='border-left:4px solid #2f855a; padding:10px; background-color:#f0f8f0'>
                            <h3 style='margin:0; color:#2f855a'>TOP: {c['pitch']}</h3>
                            <p style='margin:5px 0; font-size:20px'><strong>{c.get('pct','?')}%</strong> recommendation</p>
                            <p style='margin:0; font-size:14px; color:#666'>Expected after RE: <strong>{after_re_val:.3f}</strong></p>
                            <p style='margin:0; font-size:12px; color:#888'>Run value: {run_val:+.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Convert to float safely
                        try:
                            after_re_val = float(c.get('expected_after_re', 0))
                            run_val = float(c.get('expected_run_value', 0))
                        except (ValueError, TypeError):
                            after_re_val = 0.0
                            run_val = 0.0
                        
                        st.markdown(f"""
                        <div style='padding:10px; border-left:2px solid #ddd'>
                            <h4 style='margin:0'>{c['pitch']}</h4>
                            <p style='margin:5px 0'><strong>{c.get('pct','?')}%</strong></p>
                            <p style='margin:0; font-size:14px; color:#666'>Expected after RE: {after_re_val:.3f}</p>
                            <p style='margin:0; font-size:12px; color:#888'>Run value: {run_val:+.3f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with right:
                    loc = c.get('location', '')
                    cx, cy = _loc_to_svg_coords(loc)
                    r, color = _prob_to_visuals(c.get('pct', 0))
                    
                    # Convert to float safely for SVG title
                    try:
                        after_re_val = float(c.get('expected_after_re', 0))
                    except (ValueError, TypeError):
                        after_re_val = 0.0
                    
                    svg = f'''<svg width="200" height="220" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                        <title>{c.get('pitch')} — Expected after RE: {after_re_val:.3f}</title>
                        <rect x="15" y="8" width="70" height="84" fill="#ffffff" stroke="#000" rx="4" />
                        <text x="50" y="20" font-size="4" text-anchor="middle" fill="#666">{loc}</text>
                        <circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="0.9" />
                        <text x="50" y="98" font-size="5" text-anchor="middle">{c.get('pitch')} {c.get('pct','?')}%</text>
                    </svg>'''
                    
                    if is_top:
                        st.markdown(f"<div style='border:2px solid #2f855a; display:inline-block; padding:6px; border-radius:4px'>{svg}</div>", unsafe_allow_html=True)
                    else:
                        st.write(svg, unsafe_allow_html=True)
                
                st.markdown("---")

    st.subheader("Record Pitch Result")
    
    if candidates and any('pct' in c for c in candidates):
        score_map = {c['pitch']: c.get('pct', 0) for c in candidates}
        arsenal_list = sorted(arsenal_list, key=lambda p: -score_map.get(p, 0))

    with st.form(key='select_next'):
        st.selectbox("Select pitch thrown", options=arsenal_list, key='selected_pitch')
        st.selectbox("Outcome", ["Ball", "Called Strike", "Swinging Strike", "Foul", 
                                 "In play - out", "In play - single", "In play - double", 
                                 "In play - triple", "In play - home run", "Other (end)"], 
                    key='selected_outcome')
        st.checkbox("At-bat continues", value=True, key='selected_continue_ab')
        submit_pitch = st.form_submit_button("Submit pitch")

    if submit_pitch:
        selected_pitch = st.session_state.get('selected_pitch')
        outcome = st.session_state.get('selected_outcome')
        continue_ab = st.session_state.get('selected_continue_ab', True)

        rec = {"pitch": selected_pitch, "outcome": outcome, "count_before": count_input}
        st.session_state.setdefault("history", [])
        st.session_state["history"].append(rec)

        b, s = 0, 0
        if '-' in count_input:
            try:
                parts = count_input.split('-')
                b = int(parts[0])
                s = int(parts[1])
            except Exception:
                pass

        if outcome == "Ball":
            b = min(4, b + 1)
        elif outcome in ("Called Strike", "Swinging Strike"):
            s = min(3, s + 1)
        elif outcome == "Foul":
            if s < 2:
                s = min(2, s + 1)
        elif outcome.startswith("In play"):
            continue_ab = False

        if b >= 4:
            continue_ab = False
            st.success("Walk (4 balls)")
        if s >= 3:
            continue_ab = False
            st.success("Strikeout (3 strikes)")

        st.session_state["count"] = f"{b}-{s}"
        st.session_state["atbat_active"] = bool(continue_ab)

        if outcome.startswith("In play"):
            if "out" in outcome:
                st.session_state["situation"]["outs"] = min(2, st.session_state["situation"].get("outs", 0) + 1)
            else:
                if "single" in outcome:
                    st.session_state["situation"]["on_1b"] = True
                elif "double" in outcome:
                    st.session_state["situation"]["on_2b"] = True
                elif "home run" in outcome:
                    st.session_state["situation"]["on_1b"] = False
                    st.session_state["situation"]["on_2b"] = False
                    st.session_state["situation"]["on_3b"] = False
        
        st.success(f"Recorded: {selected_pitch} -> {outcome}")
        st.info(f"Count now: **{st.session_state['count']}**")
        st.rerun()

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
    st.download_button("Download at-bat summary", 
                      data=_json.dumps(atbat_summary, indent=2), 
                      file_name="atbat_summary.json", 
                      mime="application/json")