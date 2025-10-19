"""Build pitcher arsenals from raw Statcast CSV data.

This script reads a statcast CSV (default: data/raw/statcast_2025.csv),
extracts pitcher id and pitch_type, groups by pitcher, and writes
pitcher_assets/pitcher_arsenals.csv with rows: pitcher,pitch_type

It will back up the existing `pitcher_assets/pitcher_arsenals.csv` to
`pitcher_assets/pitcher_arsenals.csv.bak` before overwriting.
"""
from pathlib import Path
import pandas as pd
import shutil
import argparse

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STATCAST = ROOT / "data" / "raw" / "statcast_2025.csv"
OUT_DIR = ROOT / "pitcher_assets"
OUT_FILE = OUT_DIR / "pitcher_arsenals.csv"
BACKUP_FILE = OUT_DIR / "pitcher_arsenals.csv.bak"


def build(statcast_path: Path):
    statcast_path = Path(statcast_path)
    if not statcast_path.exists():
        raise FileNotFoundError(f"Statcast CSV not found at {statcast_path}")

    print(f"Reading statcast file: {statcast_path}")
    df = pd.read_csv(statcast_path)

    # Common pitch type column names: 'pitch_type', 'pitch_type_name', 'pitch', etc.
    pitch_col = None
    for c in ["pitch_type", "pitch_type_name", "pitch", "pitches"]:
        if c in df.columns:
            pitch_col = c
            break
    if pitch_col is None:
        raise ValueError("Could not find pitch type column in statcast file. Found columns: " + ",".join(df.columns))

    # Common pitcher id columns: 'pitcher', 'pitcher_id', 'pitcher_id_mlb'
    pid_col = None
    for c in ["pitcher", "pitcher_id", "pitcher_id_mlb", "pitcherId"]:
        if c in df.columns:
            pid_col = c
            break
    if pid_col is None:
        raise ValueError("Could not find pitcher id column in statcast file. Found columns: " + ",".join(df.columns))

    # Extract unique pitch types per pitcher
    grouped = df[[pid_col, pitch_col]].dropna()
    grouped[pid_col] = grouped[pid_col].astype(int)
    grouped[pitch_col] = grouped[pitch_col].astype(str)

    arsenal = grouped.drop_duplicates().sort_values([pid_col, pitch_col])

    # Ensure output dir exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Backup existing file if present
    if OUT_FILE.exists():
        print(f"Backing up existing {OUT_FILE} to {BACKUP_FILE}")
        shutil.copy2(OUT_FILE, BACKUP_FILE)

    # Write out CSV with header pitcher,pitch_type
    arsenal.to_csv(OUT_FILE, index=False, header=["pitcher", "pitch_type"] )
    print(f"Wrote pitcher arsenals to {OUT_FILE} ({len(arsenal)} rows)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--statcast", default=str(DEFAULT_STATCAST), help="Path to statcast CSV")
    args = parser.parse_args()
    build(Path(args.statcast))
