"""
Build pitcher profiles with arsenal characteristics.

For each pitcher, captures:
- Which pitches they throw
- Average velocity, movement, spin for each pitch
- Usage rates
- Effectiveness metrics

Usage:
    python src/build_pitcher_profiles.py --input data/raw/statcast_2024.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    DATA_RAW, DATA_PROCESSED,
    get_pitch_category, is_swing, is_contact,
    safe_divide
)


def build_pitcher_profiles(df: pd.DataFrame, min_pitches_per_type: int = 20) -> pd.DataFrame:
    """
    Build pitcher profiles from pitch-level data.
    
    Args:
        df: Statcast DataFrame
        min_pitches_per_type: Minimum pitches to include a pitch type in arsenal
    
    Returns:
        DataFrame with one row per (pitcher, pitch_type)
    """
    print("Building pitcher profiles...")
    
    df = df.copy()
    df = df[df["pitch_type"].notna()]
    df = df[df["pitcher"].notna()]
    
    # Add derived columns
    df["pitch_category"] = df["pitch_type"].apply(get_pitch_category)
    df["is_swing"] = df["description"].apply(is_swing)
    df["is_contact"] = df["description"].apply(is_contact)
    df["is_whiff"] = df["is_swing"] & ~df["is_contact"]
    
    # Determine if strike (called, swinging, or foul)
    df["is_strike"] = df["description"].str.contains(
        "strike|foul", case=False, na=False
    )
    
    # Group by pitcher and pitch type
    grouped = df.groupby(["pitcher", "p_throws", "pitch_type"])
    
    agg = grouped.agg(
        total_pitches=("pitch_type", "count"),
        
        # Velocity and movement
        avg_velo=("release_speed", "mean"),
        std_velo=("release_speed", "std"),
        avg_spin=("release_spin_rate", "mean"),
        avg_pfx_x=("pfx_x", "mean"),  # Horizontal movement
        avg_pfx_z=("pfx_z", "mean"),  # Vertical movement
        
        # Effectiveness
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        strikes=("is_strike", "sum"),
        
        # Average location
        avg_plate_x=("plate_x", "mean"),
        avg_plate_z=("plate_z", "mean"),
    ).reset_index()
    
    # Filter to pitch types with enough usage
    agg = agg[agg["total_pitches"] >= min_pitches_per_type]
    
    # Calculate rates
    agg["whiff_rate"] = agg.apply(
        lambda r: safe_divide(r["whiffs"], r["swings"]), axis=1
    )
    agg["strike_rate"] = agg.apply(
        lambda r: safe_divide(r["strikes"], r["total_pitches"]), axis=1
    )
    
    # Add pitch category
    agg["pitch_category"] = agg["pitch_type"].apply(get_pitch_category)
    
    # Calculate usage rate within pitcher's arsenal
    pitcher_totals = agg.groupby("pitcher")["total_pitches"].transform("sum")
    agg["usage_rate"] = agg["total_pitches"] / pitcher_totals
    
    # Add velocity differential from fastball
    fb_velos = agg[agg["pitch_category"] == "FB"].groupby("pitcher")["avg_velo"].max()
    agg["fb_velo"] = agg["pitcher"].map(fb_velos)
    agg["velo_diff_from_fb"] = agg["fb_velo"] - agg["avg_velo"]
    
    print(f"Built profiles for {agg['pitcher'].nunique()} pitchers")
    print(f"Total pitch-type combinations: {len(agg)}")
    
    return agg


def build_arsenal_summary(profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Create a summary of each pitcher's arsenal.
    
    Returns one row per pitcher with:
    - List of pitch types
    - Primary fastball velocity
    - Number of pitch types
    - Best secondary pitch
    """
    summaries = []
    
    for pitcher, group in profiles.groupby("pitcher"):
        # Sort by usage
        group = group.sort_values("usage_rate", ascending=False)
        
        # Get pitch types
        pitch_types = group["pitch_type"].tolist()
        
        # Get fastball info
        fb_group = group[group["pitch_category"] == "FB"]
        primary_fb = fb_group.iloc[0]["pitch_type"] if len(fb_group) > 0 else None
        fb_velo = fb_group["avg_velo"].max() if len(fb_group) > 0 else None
        
        # Get best secondary pitch (highest whiff rate among non-FB)
        secondary_group = group[group["pitch_category"] != "FB"]
        if len(secondary_group) > 0:
            best_secondary = secondary_group.sort_values("whiff_rate", ascending=False).iloc[0]
            best_secondary_type = best_secondary["pitch_type"]
            best_secondary_whiff = best_secondary["whiff_rate"]
        else:
            best_secondary_type = None
            best_secondary_whiff = None
        
        summaries.append({
            "pitcher": pitcher,
            "p_throws": group.iloc[0]["p_throws"],
            "num_pitch_types": len(pitch_types),
            "pitch_types": ",".join(pitch_types),
            "primary_fb": primary_fb,
            "fb_velo": fb_velo,
            "best_secondary": best_secondary_type,
            "best_secondary_whiff": best_secondary_whiff,
        })
    
    return pd.DataFrame(summaries)


def main(input_path: Path, output_path: Path = None, summary_path: Path = None):
    """Main function to build pitcher profiles."""
    
    # Load data
    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} pitches")
    
    # Build profiles
    profiles = build_pitcher_profiles(df)
    
    # Save detailed profiles
    if output_path is None:
        output_path = DATA_PROCESSED / "pitcher_profiles.parquet"
    
    profiles.to_parquet(output_path, index=False)
    print(f"Saved profiles to {output_path}")
    
    # Build and save arsenal summary
    if summary_path is None:
        summary_path = DATA_PROCESSED / "pitcher_arsenals.parquet"
    
    summary = build_arsenal_summary(profiles)
    summary.to_parquet(summary_path, index=False)
    print(f"Saved arsenal summary to {summary_path}")
    
    # Also save CSV for easy viewing
    csv_path = summary_path.with_suffix(".csv")
    summary.to_csv(csv_path, index=False)
    print(f"Saved CSV version to {csv_path}")
    
    # Print sample
    print("\nSample pitcher profiles:")
    sample_pitchers = profiles["pitcher"].unique()[:2]
    print(profiles[profiles["pitcher"].isin(sample_pitchers)].to_string())
    
    print("\nSample arsenal summary:")
    print(summary.head(10).to_string())
    
    return profiles, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pitcher profiles")
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
