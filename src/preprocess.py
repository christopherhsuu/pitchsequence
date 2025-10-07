import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KEEP = [
    "game_date","pitcher","batter","inning","at_bat_number","pitch_number",
    "balls","strikes","outs_when_up","on_1b","on_2b","on_3b",
    "stand","p_throws","pitch_type","release_speed","release_spin_rate",
    "zone","plate_x","plate_z","description","events","runs_scored"
]

def load_and_clean(year: int) -> pd.DataFrame:
    path = RAW_DIR / f"statcast_{year}.csv"
    df = pd.read_csv(path, low_memory=False)

    if "runs_scored" not in df.columns:
        run_map = {
            "home_run": 1,
            "triple": 1,
            "double": 1,
            "single": 1,
            "walk": 0,
            "intent_walk": 0,
            "strikeout": 0,
            "field_out": 0,
            "grounded_into_double_play": 0,
            "hit_by_pitch": 0
        }
        df["runs_scored"] = df["events"].map(run_map).fillna(0)

    df = df[df["pitch_type"].notna()]
    df = df[KEEP]
    for b in ["on_1b","on_2b","on_3b"]:
        df[b] = df[b].notna().astype(int)
    df["runs_scored"] = df["runs_scored"].fillna(0)
    return df


def merge_years(years):
    dfs = [load_and_clean(y) for y in years]
    return pd.concat(dfs, ignore_index=True)

if __name__ == "__main__":
    df = merge_years([2025])
    df.to_parquet(PROCESSED_DIR / "cleaned.parquet", index=False)