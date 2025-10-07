import pandas as pd
import joblib
from pathlib import Path

ARTIFACTS_DIR = Path("artifacts")
ASSETS_DIR = Path("pitcher_assets")

MODEL_PATH = ARTIFACTS_DIR / "runvalue_model.pkl"
ARSENALS_PATH = ASSETS_DIR / "pitcher_arsenals.csv"

def load_model():
    return joblib.load(MODEL_PATH)

def load_arsenals():
    return pd.read_csv(ARSENALS_PATH)

def recommend_next_pitch(state: dict, pitcher_id: int):
    model = load_model()
    ars = load_arsenals()
    pitches = ars[ars["pitcher"] == pitcher_id]["pitch_type"].unique().tolist()

    # All numeric and categorical columns used in training
    all_cols = [
        "balls","strikes","outs_when_up",
        "on_1b","on_2b","on_3b",
        "release_speed","release_spin_rate",
        "zone","plate_x","plate_z","last_pitch_speed_delta",
        "batter_pitchtype_woba","batter_pitchtype_whiff_rate","batter_pitchtype_run_value",
        "stand","p_throws","pitch_type","last_pitch_type","last_pitch_result"
    ]

    rows = []
    for pitch in pitches:
        row = state.copy()
        row["pitch_type"] = pitch
        for k in ["batter_pitchtype_woba","batter_pitchtype_whiff_rate","batter_pitchtype_run_value"]:
            row.setdefault(k, 0.0)
        rows.append(row)

    X = pd.DataFrame(rows, columns=all_cols)
    preds = model.predict(X)
    best_idx = preds.argmin()
    return pitches[best_idx], float(preds[best_idx]), dict(zip(pitches, preds))

