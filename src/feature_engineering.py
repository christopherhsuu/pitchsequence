import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["game_date","pitcher","at_bat_number","pitch_number"])
    df["last_pitch_type"] = df.groupby(["pitcher","at_bat_number"])["pitch_type"].shift(1)
    df["last_pitch_result"] = df.groupby(["pitcher","at_bat_number"])["description"].shift(1)
    df["last_pitch_speed_delta"] = df.groupby(["pitcher","at_bat_number"])["release_speed"].diff(1)
    return df

def add_matchup_features(df: pd.DataFrame) -> pd.DataFrame:
    if "estimated_woba_using_speedangle" not in df.columns:
        df["estimated_woba_using_speedangle"] = 0.0

    grouped = df.groupby(["batter", "pitch_type", "p_throws"])
    stats = grouped.agg({
        "estimated_woba_using_speedangle": "mean",
        "description": lambda x: (x.isin(["swinging_strike", "swinging_strike_blocked"]).sum() / len(x)),
        "run_value": "mean"
    }).reset_index()
    stats = stats.rename(columns={
        "estimated_woba_using_speedangle": "batter_pitchtype_woba",
        "description": "batter_pitchtype_whiff_rate",
        "run_value": "batter_pitchtype_run_value"
    })
    df = df.merge(stats, on=["batter", "pitch_type", "p_throws"], how="left")
    return df


if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "runvalue.parquet")
    df = add_sequence_features(df)
    df = add_matchup_features(df)
    df = df.dropna(subset=["pitch_type","last_pitch_type"])
    df.to_parquet(PROCESSED_DIR / "features.parquet", index=False)
