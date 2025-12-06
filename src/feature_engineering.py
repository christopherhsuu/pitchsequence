"""
Feature engineering for pitch recommendation model.

This script joins:
- Pitch-level data (with run values)
- Batter vulnerability profiles
- Pitcher arsenal profiles

And creates the final feature set for model training.

Usage:
    python src/feature_engineering.py
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    DATA_PROCESSED, ARTIFACTS,
    get_pitch_category, LEAGUE_AVERAGES,
    is_in_zone
)


def load_profiles():
    """Load batter and pitcher profiles."""
    batter_path = DATA_PROCESSED / "batter_profiles_wide.parquet"
    pitcher_path = DATA_PROCESSED / "pitcher_profiles.parquet"
    batter_zone_path = DATA_PROCESSED / "batter_zone_profiles.parquet"

    if not batter_path.exists():
        raise FileNotFoundError(f"Batter profiles not found: {batter_path}")
    if not pitcher_path.exists():
        raise FileNotFoundError(f"Pitcher profiles not found: {pitcher_path}")

    batter_profiles = pd.read_parquet(batter_path)
    pitcher_profiles = pd.read_parquet(pitcher_path)

    # Load zone profiles if available
    batter_zone_profiles = None
    if batter_zone_path.exists():
        batter_zone_profiles = pd.read_parquet(batter_zone_path)
        print(f"Loaded {len(batter_zone_profiles)} batter zone profiles")
    else:
        print(f"Warning: Batter zone profiles not found at {batter_zone_path}")

    print(f"Loaded {len(batter_profiles)} batter profiles")
    print(f"Loaded {len(pitcher_profiles)} pitcher-pitch profiles")

    return batter_profiles, pitcher_profiles, batter_zone_profiles


def join_batter_features(df: pd.DataFrame, batter_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Join batter vulnerability features to pitch data.
    
    Creates features like:
    - batter_whiff_rate_vs_this_cat: Batter's whiff rate vs this pitch's category
    - batter_chase_rate_vs_this_cat: Batter's chase rate vs this pitch's category
    - etc.
    """
    print("Joining batter features...")
    
    # Add pitch category
    if "pitch_category" not in df.columns:
        df["pitch_category"] = df["pitch_type"].apply(get_pitch_category)
    
    # Join batter profiles
    df = df.merge(batter_profiles, on="batter", how="left")
    
    # Create "vs this pitch category" features
    # These pull the correct stat based on what pitch is being thrown
    def get_batter_stat_for_pitch(row, stat):
        cat = row.get("pitch_category", "FB")
        col = f"{stat}_vs_{cat}"
        if col in row.index and pd.notna(row[col]):
            return row[col]
        # Fallback to league average
        return LEAGUE_AVERAGES.get(cat, LEAGUE_AVERAGES["FB"]).get(stat, 0.0)
    
    stats_to_map = ["whiff_rate", "chase_rate", "zone_contact_rate", "gb_rate", "xwoba"]
    
    for stat in stats_to_map:
        df[f"batter_{stat}_vs_this_cat"] = df.apply(
            lambda r: get_batter_stat_for_pitch(r, stat), axis=1
        )
    
    # Also keep the raw splits for interpretability
    # (already joined from batter_profiles)
    
    return df


def join_pitcher_features(df: pd.DataFrame, pitcher_profiles: pd.DataFrame) -> pd.DataFrame:
    """
    Join pitcher arsenal features to pitch data.
    
    Creates features like:
    - pitcher_velo_this_pitch: Pitcher's typical velocity for this pitch type
    - pitcher_whiff_rate_this_pitch: Pitcher's whiff rate with this pitch
    - etc.
    """
    print("Joining pitcher features...")
    
    # Rename pitcher profile columns to avoid conflicts
    profile_cols = ["avg_velo", "std_velo", "avg_spin", "avg_pfx_x", "avg_pfx_z",
                   "whiff_rate", "strike_rate", "usage_rate", "velo_diff_from_fb"]
    
    pitcher_profiles = pitcher_profiles.rename(columns={
        col: f"pitcher_{col}" for col in profile_cols if col in pitcher_profiles.columns
    })
    
    # Join on pitcher and pitch_type
    df = df.merge(
        pitcher_profiles[["pitcher", "pitch_type", "pitch_category"] + 
                        [c for c in pitcher_profiles.columns if c.startswith("pitcher_")]],
        on=["pitcher", "pitch_type"],
        how="left",
        suffixes=("", "_profile")
    )
    
    # Handle pitch_category collision
    if "pitch_category_profile" in df.columns:
        df = df.drop(columns=["pitch_category_profile"])

    return df


def add_location_features(df: pd.DataFrame, batter_zone_profiles: pd.DataFrame = None) -> pd.DataFrame:
    """
    Add pitch location features.

    Creates features like:
    - plate_x, plate_z (raw coordinates)
    - zone (1-14 grid)
    - Discretized location categories
    - Batter performance at this zone (from zone profiles)
    """
    print("Adding location features...")

    # Ensure location columns exist
    if "plate_x" not in df.columns or "plate_z" not in df.columns:
        print("Warning: plate_x/plate_z not found, skipping location features")
        return df

    # Discretize horizontal location
    df["loc_horizontal"] = pd.cut(
        df["plate_x"],
        bins=[-5, -1.0, -0.3, 0.3, 1.0, 5],
        labels=["inside_inside", "inside", "middle", "outside", "outside_outside"]
    )

    # Discretize vertical location
    df["loc_vertical"] = pd.cut(
        df["plate_z"],
        bins=[0, 1.8, 2.5, 3.5, 6],
        labels=["low", "mid_low", "mid_high", "high"]
    )

    # Is pitch in strike zone?
    if "zone" in df.columns:
        df["in_zone"] = df["zone"].apply(lambda z: 1 if pd.notna(z) and 1 <= z <= 9 else 0)

    # Join batter zone-specific stats if available
    if batter_zone_profiles is not None and "zone" in df.columns:
        # Join all zone profile data
        df = df.merge(batter_zone_profiles, on="batter", how="left")

        # Create "at this zone" features that look up the stat for the actual zone
        df["batter_swing_rate_at_zone"] = df.apply(
            lambda r: r.get(f"zone_{int(r['zone'])}_swing_rate", 0.5) if pd.notna(r.get("zone")) and 1 <= r.get("zone", 0) <= 14 else 0.5,
            axis=1
        )
        df["batter_whiff_rate_at_zone"] = df.apply(
            lambda r: r.get(f"zone_{int(r['zone'])}_whiff_rate", 0.25) if pd.notna(r.get("zone")) and 1 <= r.get("zone", 0) <= 14 else 0.25,
            axis=1
        )
        df["batter_contact_rate_at_zone"] = df.apply(
            lambda r: r.get(f"zone_{int(r['zone'])}_contact_rate", 0.75) if pd.notna(r.get("zone")) and 1 <= r.get("zone", 0) <= 14 else 0.75,
            axis=1
        )
        df["batter_xwoba_at_zone"] = df.apply(
            lambda r: r.get(f"zone_{int(r['zone'])}_xwoba", 0.32) if pd.notna(r.get("zone")) and 1 <= r.get("zone", 0) <= 14 else 0.32,
            axis=1
        )

        print(f"Added batter zone-specific features")

    print(f"Added location features (horizontal, vertical, zone-based)")

    return df


def add_situational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add game situation features."""
    print("Adding situational features...")
    
    # Platoon advantage (same hand = advantage for pitcher)
    if "stand" in df.columns and "p_throws" in df.columns:
        df["same_hand"] = (df["stand"] == df["p_throws"]).astype(int)
        df["platoon_advantage"] = (df["stand"] != df["p_throws"]).astype(int)
    
    # Count features
    if "balls" in df.columns and "strikes" in df.columns:
        df["count_balls"] = df["balls"].fillna(0).astype(int)
        df["count_strikes"] = df["strikes"].fillna(0).astype(int)
        
        # Count state categories
        df["is_ahead_in_count"] = (df["strikes"] > df["balls"]).astype(int)
        df["is_behind_in_count"] = (df["balls"] > df["strikes"]).astype(int)
        df["is_two_strikes"] = (df["strikes"] == 2).astype(int)
        df["is_three_balls"] = (df["balls"] == 3).astype(int)
        df["is_full_count"] = ((df["balls"] == 3) & (df["strikes"] == 2)).astype(int)
        df["is_first_pitch"] = ((df["balls"] == 0) & (df["strikes"] == 0)).astype(int)
    
    # Base situation
    if all(c in df.columns for c in ["on_1b", "on_2b", "on_3b"]):
        df["runners_on"] = (df["on_1b"] + df["on_2b"] + df["on_3b"]).clip(0, 3)
        df["risp"] = ((df["on_2b"] == 1) | (df["on_3b"] == 1)).astype(int)
        df["bases_empty"] = (df["runners_on"] == 0).astype(int)
        df["bases_loaded"] = (df["runners_on"] == 3).astype(int)
    
    # Outs
    if "outs_when_up" in df.columns:
        df["outs"] = df["outs_when_up"].fillna(0).astype(int).clip(0, 2)
        df["is_two_outs"] = (df["outs"] == 2).astype(int)
    
    # Score differential (if available)
    if all(c in df.columns for c in ["bat_score", "fld_score"]):
        df["score_diff"] = df["fld_score"] - df["bat_score"]  # Positive = pitcher's team ahead
        df["is_close_game"] = (abs(df["score_diff"]) <= 2).astype(int)
    elif all(c in df.columns for c in ["home_score", "away_score", "inning_topbot"]):
        df["score_diff"] = df.apply(
            lambda r: r["home_score"] - r["away_score"] if r.get("inning_topbot") == "Bot" 
                     else r["away_score"] - r["home_score"],
            axis=1
        )
        df["is_close_game"] = (abs(df["score_diff"]) <= 2).astype(int)
    
    # Inning
    if "inning" in df.columns:
        df["inning_num"] = df["inning"].fillna(1).astype(int)
        df["is_late_inning"] = (df["inning_num"] >= 7).astype(int)
    
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between batter vulnerability and pitch characteristics."""
    print("Adding interaction features...")
    
    # Batter vulnerability * pitcher effectiveness
    if "batter_whiff_rate_vs_this_cat" in df.columns and "pitcher_whiff_rate" in df.columns:
        df["whiff_interaction"] = (
            df["batter_whiff_rate_vs_this_cat"] * df["pitcher_whiff_rate"]
        ).fillna(0)
    
    if "batter_chase_rate_vs_this_cat" in df.columns:
        # Chase opportunity (more valuable when ahead in count)
        df["chase_opportunity"] = df["batter_chase_rate_vs_this_cat"] * df.get("is_ahead_in_count", 0)
    
    # Velocity differential impact
    if "pitcher_velo_diff_from_fb" in df.columns:
        df["speed_change_magnitude"] = abs(df["pitcher_velo_diff_from_fb"]).fillna(0)
    
    # Count-based aggression (batters swing more when behind, pitchers can exploit)
    if "is_two_strikes" in df.columns and "batter_whiff_rate_vs_this_cat" in df.columns:
        df["two_strike_whiff_boost"] = (
            df["is_two_strikes"] * df["batter_whiff_rate_vs_this_cat"]
        )
    
    return df


def select_final_features(df: pd.DataFrame) -> tuple:
    """
    Select features for model training.
    
    Returns:
        (feature_df, target, feature_names)
    """
    print("Selecting final features...")
    
    # Define feature groups
    count_features = [
        "count_balls", "count_strikes",
        "is_ahead_in_count", "is_behind_in_count",
        "is_two_strikes", "is_three_balls", "is_full_count", "is_first_pitch"
    ]
    
    situation_features = [
        "outs", "is_two_outs",
        "on_1b", "on_2b", "on_3b",
        "runners_on", "risp", "bases_empty", "bases_loaded",
        "is_late_inning", "is_close_game"
    ]
    
    matchup_features = [
        "same_hand", "platoon_advantage"
    ]
    
    batter_features = [
        "batter_whiff_rate_vs_this_cat",
        "batter_chase_rate_vs_this_cat", 
        "batter_zone_contact_rate_vs_this_cat",
        "batter_gb_rate_vs_this_cat",
        "batter_xwoba_vs_this_cat",
        # Raw splits (all categories)
        "whiff_rate_vs_FB", "whiff_rate_vs_BR", "whiff_rate_vs_OS",
        "chase_rate_vs_FB", "chase_rate_vs_BR", "chase_rate_vs_OS",
        "xwoba_vs_FB", "xwoba_vs_BR", "xwoba_vs_OS",
    ]
    
    pitcher_features = [
        "pitcher_avg_velo", "pitcher_avg_spin",
        "pitcher_avg_pfx_x", "pitcher_avg_pfx_z",
        "pitcher_whiff_rate", "pitcher_strike_rate",
        "pitcher_usage_rate", "pitcher_velo_diff_from_fb"
    ]
    
    sequence_features = [
        "pitch_num_in_ab",
        "same_as_last", "consecutive_same_pitch",
        "speed_delta"
    ]
    
    interaction_features = [
        "whiff_interaction", "chase_opportunity",
        "speed_change_magnitude", "two_strike_whiff_boost"
    ]
    
    location_features = [
        "plate_x", "plate_z",
        "zone", "in_zone",
        # Batter performance at this specific zone
        "batter_swing_rate_at_zone",
        "batter_whiff_rate_at_zone",
        "batter_contact_rate_at_zone",
        "batter_xwoba_at_zone"
    ]

    categorical_features = [
        "pitch_type", "pitch_category",
        "last_pitch_type", "last_pitch_category",
        "stand", "p_throws",
        "loc_horizontal", "loc_vertical"  # Location categories (categorical)
    ]

    # Combine all features
    all_features = (
        count_features + situation_features + matchup_features +
        batter_features + pitcher_features + sequence_features +
        interaction_features + location_features + categorical_features
    )
    
    # Filter to available columns
    available = [f for f in all_features if f in df.columns]
    missing = [f for f in all_features if f not in df.columns]
    
    if missing:
        print(f"Note: {len(missing)} features not available: {missing[:10]}...")
    
    print(f"Using {len(available)} features")
    
    # Extract feature matrix and target
    feature_df = df[available].copy()
    target = df["run_value"].copy() if "run_value" in df.columns else None
    
    return feature_df, target, available


def main():
    """Main feature engineering pipeline."""
    
    # Load pitch data
    pitch_path = DATA_PROCESSED / "pitches_with_rv.parquet"
    if not pitch_path.exists():
        print(f"Error: Preprocessed pitch data not found: {pitch_path}")
        print("Run preprocess.py first.")
        return
    
    print(f"Loading pitch data from {pitch_path}...")
    df = pd.read_parquet(pitch_path)
    print(f"Loaded {len(df):,} pitches")
    
    # Load profiles
    batter_profiles, pitcher_profiles, batter_zone_profiles = load_profiles()

    # Join features
    df = join_batter_features(df, batter_profiles)
    df = join_pitcher_features(df, pitcher_profiles)
    df = add_location_features(df, batter_zone_profiles)
    df = add_situational_features(df)
    df = add_interaction_features(df)

    # Select final features
    feature_df, target, feature_names = select_final_features(df)
    
    # Save full dataset with features
    output_path = DATA_PROCESSED / "features.parquet"
    
    # Combine features with identifiers and target
    id_cols = ["game_pk", "game_date", "pitcher", "batter", "at_bat_number", "pitch_number"]
    id_cols = [c for c in id_cols if c in df.columns]
    
    output_df = df[id_cols].copy()
    output_df = pd.concat([output_df, feature_df], axis=1)
    if target is not None:
        output_df["run_value"] = target
    
    output_df.to_parquet(output_path, index=False)
    print(f"Saved features to {output_path}")
    
    # Save feature column list
    import json
    feature_path = ARTIFACTS / "feature_columns.json"
    with open(feature_path, "w") as f:
        json.dump(feature_names, f, indent=2)
    print(f"Saved feature list to {feature_path}")
    
    # Print summary
    print("\nFeature summary:")
    print(f"  Total samples: {len(output_df):,}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Target (run_value) mean: {target.mean():.4f}" if target is not None else "  No target")
    
    # Show sample
    print("\nSample features:")
    print(feature_df.head(3).T)
    
    return output_df, feature_names


if __name__ == "__main__":
    main()
