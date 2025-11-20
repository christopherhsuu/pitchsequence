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
    best_idx = int(preds.argmin())

    # return 4 values for callers that expect debug info as the 4th element
    preds_dict = dict(zip(pitches, preds))
    debug_rows = X.to_dict(orient="records")
    return pitches[best_idx], float(preds[best_idx]), preds_dict, debug_rows


def get_cluster_features(cluster_id):
    """Return cluster-specific feature adjustments.

    This is a small, tolerant helper that attempts to load cluster-based
    adjustments from disk if present, otherwise returns None. The app
    treats ``None`` as "no adjustment".
    """
    try:
        # optional artifact: artifacts/cluster_pitch_adjustments.csv
        p = ARTIFACTS_DIR / "cluster_pitch_adjustments.csv"
        if p.exists():
            df = pd.read_csv(p)
            # expect columns: cluster, pitch_type, multiplier
            sub = df[df["cluster"] == cluster_id]
            if sub.empty:
                return None
            return {r["pitch_type"]: float(r.get("multiplier", 1.0)) for _, r in sub.iterrows()}
    except Exception:
        # be tolerant; callers handle None
        return None


def adjust_pitch_recommendation(prob_dict, cluster_feats=None):
    """Adjust pitch probabilities based on cluster features.

    prob_dict: mapping pitch_type -> probability
    cluster_feats: None or mapping pitch_type->multiplier

    If cluster_feats is None, returns prob_dict unchanged. Otherwise
    multiplies each probability by the given multiplier (default 1.0)
    and re-normalizes to sum to 1.0.
    """
    if not prob_dict:
        return prob_dict
    if not cluster_feats:
        return prob_dict

    # apply multipliers
    adjusted = {}
    for p, prob in prob_dict.items():
        mult = float(cluster_feats.get(p, 1.0)) if isinstance(cluster_feats, dict) else 1.0
        adjusted[p] = prob * mult

    total = sum(adjusted.values())
    if total <= 0:
        # fallback to original distribution if something went wrong
        return prob_dict
    for p in adjusted:
        adjusted[p] = adjusted[p] / total
    return adjusted

