import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path("data/processed")
STATIC_DIR = Path("static")

def bases_state(row):
    return f"{int(row['on_1b'])}{int(row['on_2b'])}{int(row['on_3b'])}"

def run_value(df: pd.DataFrame) -> pd.DataFrame:
    re = pd.read_csv(STATIC_DIR / "run_expectancy_24.csv")
    re_map = {(r.outs, r.bases_state): r.run_expectancy for r in re.itertuples()}
    df["before_state"] = df.apply(lambda r: re_map.get((r["outs_when_up"], bases_state(r)), 0), axis=1)
    df["after_state"] = df["before_state"].shift(-1)
    df["after_state"] = df["after_state"].fillna(0)
    df["run_value"] = df["after_state"] - df["before_state"] + df["runs_scored"]
    return df

if __name__ == "__main__":
    df = pd.read_parquet(PROCESSED_DIR / "cleaned.parquet")
    df = run_value(df)
    df.to_parquet(PROCESSED_DIR / "runvalue.parquet", index=False)
