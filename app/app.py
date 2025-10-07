import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
from predict import recommend_next_pitch, MODEL_PATH
st.set_page_config(page_title="PitchSequence", layout="wide")
st.title("PitchSequence — Optimal Next Pitch Recommendation")


with st.sidebar:
    st.header("Game Context")
    pitcher_id = st.number_input("Pitcher ID", min_value=1, value=123456)
    balls = st.selectbox("Balls", [0,1,2,3])
    strikes = st.selectbox("Strikes", [0,1,2])
    outs = st.selectbox("Outs", [0,1,2])
    on1 = st.checkbox("Runner on 1B")
    on2 = st.checkbox("Runner on 2B")
    on3 = st.checkbox("Runner on 3B")
    stand = st.selectbox("Batter Hand", ["L","R"])
    p_throws = st.selectbox("Pitcher Hand", ["L","R"])
    release_speed = st.number_input("Release Speed", value=94.0)
    spin = st.number_input("Spin Rate", value=2200)
    zone = st.number_input("Zone (1-14)", min_value=1, max_value=14, value=5)

    # sliders first
    plate_x = st.slider("plate_x", -2.0, 2.0, 0.0, 0.01)
    plate_z = st.slider("plate_z", 0.0, 5.0, 2.5, 0.01)

    # now draw figure using those values
    st.subheader("Select Pitch Location")
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_xlim(-2,2)
    ax.set_ylim(0,5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.axhline(2.5, color='gray', linewidth=0.5)
    rect = plt.Rectangle((-0.83,1.5), 1.66, 2, fill=False, color='red', linewidth=1.5)
    ax.add_patch(rect)
    ax.scatter(plate_x, plate_z, s=250, color='white', edgecolor='black', linewidth=1.5)
    st.pyplot(fig, clear_figure=False)

    last_pt = st.selectbox("Last Pitch Type", ["FF","SL","CU","CH"])
    last_res = st.selectbox("Last Pitch Result", ["ball","called_strike","swinging_strike","foul","in_play"])
    last_delta = st.number_input("Last Pitch Speed Δ", value=0.0)


state = {
    "balls": balls,
    "strikes": strikes,
    "outs_when_up": outs,
    "on_1b": int(on1),
    "on_2b": int(on2),
    "on_3b": int(on3),
    "stand": stand,
    "p_throws": p_throws,
    "release_speed": release_speed,
    "release_spin_rate": spin,
    "zone": zone,
    "plate_x": plate_x,
    "plate_z": plate_z,
    "last_pitch_type": last_pt,
    "last_pitch_result": last_res,
    "last_pitch_speed_delta": last_delta
}

if st.button("Recommend Next Pitch"):
    if not MODEL_PATH.exists():
        st.error("Model file missing. Train first.")
    else:
        pitch, value, preds = recommend_next_pitch(state, pitcher_id)
        st.subheader(f"Recommended: {pitch}  (expected run value {value:+.3f})")
        st.dataframe(pd.DataFrame({"pitch_type": list(preds.keys()), "expected_run_value": list(preds.values())}).sort_values("expected_run_value"))
