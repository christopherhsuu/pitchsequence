"""
Preprocess Statcast data and calculate run values.

This script:
1. Cleans raw Statcast data
2. Calculates run values using RE24 matrix
3. Adds sequence features within at-bats

Usage:
    python src/preprocess.py --input data/raw/statcast_2024.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    DATA_RAW, DATA_PROCESSED, STATIC,
    load_run_expectancy, bases_to_state, get_run_expectancy,
    get_pitch_category, classify_outcome,
    is_swing, is_in_zone, is_contact
)


def clean_statcast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw Statcast data.
    
    - Remove rows with missing critical data
    - Standardize column types
    - Remove duplicate pitches
    """
    print("Cleaning data...")
    initial_count = len(df)
    
    # Remove rows missing critical columns
    required_cols = ["pitcher", "batter", "pitch_type", "description"]
    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]
    
    # Convert date
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])
    
    # Ensure numeric columns are numeric
    numeric_cols = [
        "balls", "strikes", "outs_when_up", "inning",
        "on_1b", "on_2b", "on_3b",
        "release_speed", "release_spin_rate",
        "plate_x", "plate_z", "zone",
        "launch_speed", "launch_angle",
        "estimated_woba_using_speedangle"
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Convert base runners to binary (Statcast uses player IDs, we want 0/1)
    for col in ["on_1b", "on_2b", "on_3b"]:
        if col in df.columns:
            df[col] = df[col].notna().astype(int)
    
    # Remove duplicates
    if "game_pk" in df.columns and "at_bat_number" in df.columns and "pitch_number" in df.columns:
        df = df.drop_duplicates(subset=["game_pk", "at_bat_number", "pitch_number"])
    
    # Sort for sequence features
    sort_cols = ["game_date", "game_pk", "at_bat_number", "pitch_number"]
    sort_cols = [c for c in sort_cols if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols)
    
    print(f"Cleaned: {initial_count:,} -> {len(df):,} rows")
    
    return df.reset_index(drop=True)


def add_run_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate run value for each pitch using RE24 matrix.

    Run value = (RE after pitch - RE before pitch) + runs scored on play

    This version uses:
    1. Actual runs scored from post_bat_score / post_fld_score columns
    2. Next pitch's base-out state for RE_after (accounts for all baserunner movement)
    """
    print("Calculating run values...")

    re_matrix = load_run_expectancy()

    # Calculate before state
    df["bases_state"] = df.apply(
        lambda r: bases_to_state(r.get("on_1b", 0), r.get("on_2b", 0), r.get("on_3b", 0)),
        axis=1
    )
    df["outs"] = df["outs_when_up"].fillna(0).astype(int).clip(0, 2)

    df["re_before"] = df.apply(
        lambda r: re_matrix.get((r["outs"], r["bases_state"]), 0.0),
        axis=1
    )

    # Group by game and inning half for proper sequencing
    group_cols = ["game_pk", "inning", "inning_topbot"] if all(
        c in df.columns for c in ["game_pk", "inning", "inning_topbot"]
    ) else ["game_pk"]

    # Get next pitch's state (which reflects the result of THIS pitch)
    df["next_outs"] = df.groupby(group_cols)["outs"].shift(-1)
    df["next_bases"] = df.groupby(group_cols)["bases_state"].shift(-1)

    # Calculate RE after using next pitch's state
    def get_re_after(row):
        next_outs = row.get("next_outs")
        next_bases = row.get("next_bases")

        # If next outs is NaN or 0 when current is 2, inning ended
        if pd.isna(next_outs) or (row["outs"] == 2 and next_outs == 0):
            return 0.0

        # If we have 3 outs recorded in data, inning ended
        if next_outs >= 3:
            return 0.0

        # Otherwise look up RE in matrix
        if pd.notna(next_bases):
            return re_matrix.get((int(next_outs), next_bases), 0.0)
        return 0.0

    df["re_after"] = df.apply(get_re_after, axis=1)

    # Calculate ACTUAL runs scored on this pitch
    # Statcast provides post_bat_score (runs scored by batting team after pitch)
    df["runs_on_play"] = 0.0

    if "post_bat_score" in df.columns and "bat_score" in df.columns:
        # Runs scored = post_bat_score - bat_score (score before pitch)
        df["runs_on_play"] = (df["post_bat_score"] - df["bat_score"]).fillna(0.0)
    elif "post_fld_score" in df.columns and "fld_score" in df.columns:
        # If batting perspective not available, use fielding perspective
        # (fld_score is runs allowed by fielding team)
        df["runs_on_play"] = (df["post_fld_score"] - df["fld_score"]).fillna(0.0)
    elif "events" in df.columns:
        # Fallback: infer from events (less accurate)
        print("Warning: Using event-based run estimation (less accurate)")
        runs_map = {
            "home_run": 1,
            "sac_fly": 1,
        }
        for event, runs in runs_map.items():
            mask = df["events"].str.contains(event, case=False, na=False)
            df.loc[mask, "runs_on_play"] = runs

    # Calculate run value
    # Positive = bad for pitcher, negative = good for pitcher
    df["run_value"] = df["re_after"] - df["re_before"] + df["runs_on_play"]

    # Clean up temp columns
    df = df.drop(columns=["next_outs", "next_bases"], errors="ignore")

    print(f"Run value stats: mean={df['run_value'].mean():.4f}, std={df['run_value'].std():.4f}")
    print(f"  Runs scored: {df['runs_on_play'].sum():.0f} total, {(df['runs_on_play'] > 0).sum():,} scoring plays")

    return df


def add_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features that capture pitch sequencing within at-bats.
    """
    print("Adding sequence features...")
    
    # Group by at-bat
    ab_cols = ["game_pk", "at_bat_number"] if all(
        c in df.columns for c in ["game_pk", "at_bat_number"]
    ) else ["game_pk", "pitcher", "batter"]
    
    group = df.groupby(ab_cols)
    
    # Previous pitch info
    df["last_pitch_type"] = group["pitch_type"].shift(1)
    df["last_pitch_category"] = df["last_pitch_type"].apply(
        lambda x: get_pitch_category(x) if pd.notna(x) else None
    )
    
    # Previous pitch outcome
    if "description" in df.columns:
        df["last_pitch_result"] = group["description"].shift(1)
    
    # Velocity change
    if "release_speed" in df.columns:
        df["last_pitch_speed"] = group["release_speed"].shift(1)
        df["speed_delta"] = df["release_speed"] - df["last_pitch_speed"]
    
    # Was last pitch in zone?
    if "zone" in df.columns:
        df["last_pitch_zone"] = group["zone"].shift(1)
        df["last_pitch_in_zone"] = df["last_pitch_zone"].apply(is_in_zone)
    
    # Pitch number in at-bat
    df["pitch_num_in_ab"] = group.cumcount() + 1
    
    # Cumulative pitch type counts in this at-bat
    df["pitch_category"] = df["pitch_type"].apply(get_pitch_category)
    
    # Count of each category seen so far in AB
    for cat in ["FB", "BR", "OS"]:
        df[f"{cat}_count_in_ab"] = (df["pitch_category"] == cat).astype(int).groupby(
            [df[col] for col in ab_cols]
        ).cumsum()
    
    # Same pitch as last pitch?
    df["same_as_last"] = (df["pitch_type"] == df["last_pitch_type"]).astype(int)
    
    # Consecutive same pitch count
    def count_consecutive(series):
        result = []
        count = 0
        prev = None
        for val in series:
            if val == prev:
                count += 1
            else:
                count = 0
            result.append(count)
            prev = val
        return result
    
    df["consecutive_same_pitch"] = group["pitch_type"].transform(count_consecutive)
    
    print(f"Added {len([c for c in df.columns if c.startswith('last_') or c.endswith('_in_ab')])} sequence features")
    
    return df


def add_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features derived from pitch outcome."""
    print("Adding outcome features...")
    
    # Classify outcome
    df["outcome_class"] = df.apply(
        lambda r: classify_outcome(r.get("description"), r.get("events")),
        axis=1
    )
    
    # Binary flags
    df["is_swing"] = df["description"].apply(is_swing)
    df["is_in_zone"] = df["zone"].apply(is_in_zone)
    df["is_contact"] = df["description"].apply(is_contact)
    df["is_whiff"] = df["is_swing"] & ~df["is_contact"]
    df["is_chase"] = df["is_swing"] & ~df["is_in_zone"]
    df["is_called_strike"] = df["outcome_class"] == "called_strike"
    
    return df


def main(input_path: Path, output_path: Path = None):
    """Main preprocessing function."""
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} pitches")
    
    # Clean
    df = clean_statcast(df)
    
    # Add run values
    df = add_run_values(df)
    
    # Add sequence features
    df = add_sequence_features(df)
    
    # Add outcome features
    df = add_outcome_features(df)
    
    # Save
    if output_path is None:
        output_path = DATA_PROCESSED / "pitches_with_rv.parquet"
    
    df.to_parquet(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Print summary
    print("\nData summary:")
    print(f"  Total pitches: {len(df):,}")
    print(f"  Unique pitchers: {df['pitcher'].nunique():,}")
    print(f"  Unique batters: {df['batter'].nunique():,}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"\nRun value distribution:")
    print(df["run_value"].describe())
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Statcast data")
    parser.add_argument("--input", type=str, default=str(DATA_RAW / "statcast_2024.csv"),
                       help="Input Statcast CSV path")
    parser.add_argument("--output", type=str, help="Output parquet path")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)
    
    main(input_path, output_path)
