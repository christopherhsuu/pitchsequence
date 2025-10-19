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

def recommend_next_pitch(state: dict, pitcher_id: int, batter_stats_map: dict = None):
    model = load_model()
    ars = load_arsenals()
    pitches = ars[ars["pitcher"] == pitcher_id]["pitch_type"].unique().tolist()
    if not pitches:
        # fallback default arsenal
        pitches = ["FF", "SL", "CU"]

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
        # provide realistic default release speed and spin by pitch type to let model differentiate
        pt = str(pitch).upper()
        default_speed = 92.0
        default_spin = 2100.0
        if pt in ("FF", "FT", "FA", "FC"):
            default_speed = 94.5
            default_spin = 2300.0
        elif pt in ("SL", "CU", "KC", "CH"):
            default_speed = 84.0
            default_spin = 2600.0
        elif pt in ("SI", "FS"):
            default_speed = 92.0
            default_spin = 2200.0
        elif pt in ("ST", "SV", "EP"):
            default_speed = 88.0
            default_spin = 2000.0

        row.setdefault("release_speed", default_speed)
        row.setdefault("release_spin_rate", default_spin)
        # if batter_stats_map provided, set the batter-specific features for this pitch
        if batter_stats_map and pitch in batter_stats_map:
            woba, whiff, brv = batter_stats_map.get(pitch, (0.0, 0.0, 0.0))
            row["batter_pitchtype_woba"] = woba
            row["batter_pitchtype_whiff_rate"] = whiff
            row["batter_pitchtype_run_value"] = brv
        else:
            for k in ["batter_pitchtype_woba","batter_pitchtype_whiff_rate","batter_pitchtype_run_value"]:
                row.setdefault(k, 0.0)
        rows.append(row)
    if not rows:
        raise ValueError("No candidate pitches to score for pitcher_id={}".format(pitcher_id))

    X = pd.DataFrame(rows, columns=all_cols)
    if X.shape[0] == 0:
        raise ValueError("Model input X is empty — cannot predict")
    preds = model.predict(X)
    best_idx = preds.argmin()
    return pitches[best_idx], float(preds[best_idx]), dict(zip(pitches, preds))

