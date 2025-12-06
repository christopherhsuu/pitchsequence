"""
Build batter vulnerability profiles.

This script aggregates batter performance against different pitch categories
to create features that capture what pitches each batter struggles against.

Key outputs per batter per pitch category (FB/BR/OS):
- whiff_rate: How often they swing and miss
- chase_rate: How often they chase out of zone
- zone_contact_rate: How well they make contact in the zone
- xwoba: Quality of contact when they connect
- gb_rate: Ground ball tendency
- sample_size: For regression to mean

Usage:
    python src/build_batter_profiles.py --input data/raw/statcast_2024.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    DATA_RAW, DATA_PROCESSED,
    get_pitch_category, LEAGUE_AVERAGES,
    is_swing, is_in_zone, is_contact,
    regress_to_mean, safe_divide
)


def build_batter_profiles(df: pd.DataFrame, min_pitches: int = 50) -> pd.DataFrame:
    """
    Build batter vulnerability profiles from pitch-level data.
    
    Args:
        df: Statcast DataFrame with pitch-level data
        min_pitches: Minimum pitches to include a batter (before regression)
    
    Returns:
        DataFrame with one row per (batter, pitch_category)
    """
    print("Building batter profiles...")
    
    # Add pitch category
    df = df.copy()
    df["pitch_category"] = df["pitch_type"].apply(get_pitch_category)
    df = df[df["pitch_category"].isin(["FB", "BR", "OS"])]
    
    # Add derived columns
    df["is_swing"] = df["description"].apply(is_swing)
    df["is_in_zone"] = df["zone"].apply(is_in_zone)
    df["is_contact"] = df["description"].apply(is_contact)
    df["is_whiff"] = df["is_swing"] & ~df["is_contact"]
    df["is_chase"] = df["is_swing"] & ~df["is_in_zone"]
    df["is_zone_swing"] = df["is_swing"] & df["is_in_zone"]
    df["is_zone_contact"] = df["is_contact"] & df["is_in_zone"]
    
    # Determine if ball was hit on ground (launch_angle < 10)
    df["is_groundball"] = df["launch_angle"].fillna(999) < 10
    df["is_batted_ball"] = df["launch_speed"].notna()
    
    # Group by batter and pitch category
    grouped = df.groupby(["batter", "pitch_category"])
    
    # Calculate raw stats
    agg = grouped.agg(
        total_pitches=("pitch_type", "count"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        out_of_zone=("is_in_zone", lambda x: (~x).sum()),
        chases=("is_chase", "sum"),
        zone_swings=("is_zone_swing", "sum"),
        zone_contact=("is_zone_contact", "sum"),
        batted_balls=("is_batted_ball", "sum"),
        groundballs=("is_groundball", "sum"),
        xwoba_sum=("estimated_woba_using_speedangle", "sum"),
        xwoba_count=("estimated_woba_using_speedangle", "count"),
    ).reset_index()
    
    # Calculate rates
    agg["whiff_rate_raw"] = agg.apply(
        lambda r: safe_divide(r["whiffs"], r["swings"]), axis=1
    )
    agg["chase_rate_raw"] = agg.apply(
        lambda r: safe_divide(r["chases"], r["out_of_zone"]), axis=1
    )
    agg["zone_contact_rate_raw"] = agg.apply(
        lambda r: safe_divide(r["zone_contact"], r["zone_swings"]), axis=1
    )
    agg["gb_rate_raw"] = agg.apply(
        lambda r: safe_divide(r["groundballs"], r["batted_balls"]), axis=1
    )
    agg["xwoba_raw"] = agg.apply(
        lambda r: safe_divide(r["xwoba_sum"], r["xwoba_count"], default=0.320), axis=1
    )
    
    # Apply regression to mean
    print("Applying regression to mean for small samples...")
    
    def apply_regression(row):
        cat = row["pitch_category"]
        n = row["swings"]  # Use swing count for regression weight
        league = LEAGUE_AVERAGES.get(cat, LEAGUE_AVERAGES["FB"])
        
        return pd.Series({
            "whiff_rate": regress_to_mean(
                row["whiff_rate_raw"], n, league["whiff_rate"],
                reliability_threshold=100
            ),
            "chase_rate": regress_to_mean(
                row["chase_rate_raw"], row["out_of_zone"], league["chase_rate"],
                reliability_threshold=100
            ),
            "zone_contact_rate": regress_to_mean(
                row["zone_contact_rate_raw"], row["zone_swings"], league["zone_contact_rate"],
                reliability_threshold=80
            ),
            "gb_rate": regress_to_mean(
                row["gb_rate_raw"], row["batted_balls"], league["gb_rate"],
                reliability_threshold=50
            ),
            "xwoba": regress_to_mean(
                row["xwoba_raw"], row["xwoba_count"], league["xwoba"],
                reliability_threshold=50
            ),
        })
    
    regressed = agg.apply(apply_regression, axis=1)
    agg = pd.concat([agg, regressed], axis=1)
    
    # Add vulnerability score (composite measure of how hittable they find this pitch category)
    # Higher = more vulnerable to this pitch type
    agg["vulnerability_score"] = (
        (1 - agg["whiff_rate"]) * 0.3 +  # Lower whiff = better at hitting
        (1 - agg["chase_rate"]) * 0.2 +   # Lower chase = more patient
        agg["zone_contact_rate"] * 0.2 +  # Higher contact = better bat control
        agg["xwoba"] * 0.3                # Higher xwoba = better quality contact
    )
    
    # Keep relevant columns
    output_cols = [
        "batter", "pitch_category",
        "total_pitches", "swings",
        "whiff_rate", "chase_rate", "zone_contact_rate", "gb_rate", "xwoba",
        "vulnerability_score",
        # Keep raw values for debugging
        "whiff_rate_raw", "chase_rate_raw", "zone_contact_rate_raw"
    ]
    
    result = agg[output_cols].copy()
    
    print(f"Built profiles for {result['batter'].nunique()} batters")
    
    return result


def pivot_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot profiles so each batter has one row with columns for each pitch category.
    
    This makes it easier to join to pitch-level data.
    """
    # Pivot each stat
    stats = ["whiff_rate", "chase_rate", "zone_contact_rate", "gb_rate", "xwoba", 
             "vulnerability_score", "total_pitches"]
    
    pivoted = profiles.pivot(
        index="batter",
        columns="pitch_category",
        values=stats
    )
    
    # Flatten column names
    pivoted.columns = [f"{stat}_vs_{cat}" for stat, cat in pivoted.columns]
    pivoted = pivoted.reset_index()
    
    # Fill missing values with league averages
    for cat in ["FB", "BR", "OS"]:
        league = LEAGUE_AVERAGES.get(cat, LEAGUE_AVERAGES["FB"])
        for stat in ["whiff_rate", "chase_rate", "zone_contact_rate", "gb_rate", "xwoba"]:
            col = f"{stat}_vs_{cat}"
            if col in pivoted.columns:
                pivoted[col] = pivoted[col].fillna(league[stat])
    
    return pivoted


def main(input_path: Path, output_path: Path = None, output_pivoted_path: Path = None):
    """Main function to build batter profiles."""
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} pitches")
    
    # Filter to valid pitches
    df = df[df["pitch_type"].notna()]
    df = df[df["batter"].notna()]
    print(f"After filtering: {len(df):,} pitches")
    
    # Build profiles
    profiles = build_batter_profiles(df)
    
    # Save long format
    if output_path is None:
        output_path = DATA_PROCESSED / "batter_profiles.parquet"
    
    profiles.to_parquet(output_path, index=False)
    print(f"Saved profiles to {output_path}")
    
    # Save pivoted format (one row per batter)
    if output_pivoted_path is None:
        output_pivoted_path = DATA_PROCESSED / "batter_profiles_wide.parquet"
    
    pivoted = pivot_profiles(profiles)
    pivoted.to_parquet(output_pivoted_path, index=False)
    print(f"Saved pivoted profiles to {output_pivoted_path}")
    
    # Print sample
    print("\nSample profiles (first 3 batters):")
    sample_batters = profiles["batter"].unique()[:3]
    print(profiles[profiles["batter"].isin(sample_batters)].to_string())
    
    return profiles, pivoted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build batter vulnerability profiles")
    parser.add_argument("--input", type=str, default=str(DATA_RAW / "statcast_2024.csv"),
                       help="Input Statcast CSV path")
    parser.add_argument("--output", type=str, help="Output parquet path")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        print("Run fetch_statcast.py first to download the data.")
        exit(1)
    
    main(input_path, output_path)
