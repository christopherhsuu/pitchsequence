"""
Build batter zone profiles.

For each batter, calculate performance metrics by zone (1-14):
- Zones 1-9: Strike zone (3x3 grid)
- Zones 11-14: Out of zone (balls)

Metrics per zone:
- whiff_rate: Swings and misses / swings
- contact_rate: Contact / swings
- xwoba: Expected weighted on-base average
- swing_rate: Swings / pitches seen
- chase_rate (zones 11-14): Swings outside zone / pitches outside zone

Usage:
    python src/build_batter_zone_profiles.py --input data/raw/statcast_2024.csv
"""
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    DATA_RAW, DATA_PROCESSED,
    is_swing, is_contact, is_in_zone,
    LEAGUE_AVERAGES
)


def calculate_zone_stats(df: pd.DataFrame, min_pitches: int = 20) -> pd.DataFrame:
    """
    Calculate batter performance by zone with regression to mean.

    Args:
        df: Statcast data with zone column
        min_pitches: Minimum pitches for reliability weighting

    Returns:
        DataFrame with batter_id and stats for each zone
    """
    print("Calculating zone-level statistics...")

    # Add outcome flags
    df['is_swing'] = df['description'].apply(is_swing)
    df['is_contact'] = df['description'].apply(is_contact)
    df['is_in_zone'] = df['zone'].apply(is_in_zone)

    # Calculate league averages by zone for regression to mean
    league_zone_stats = {}
    for zone in range(1, 15):
        zone_df = df[df['zone'] == zone]
        if len(zone_df) > 100:
            swings = zone_df['is_swing'].sum()
            league_zone_stats[zone] = {
                'whiff_rate': zone_df['is_swing'].sum() > 0 and zone_df[zone_df['is_swing']]['is_contact'].apply(lambda x: not x).sum() / swings or 0.25,
                'contact_rate': zone_df['is_swing'].sum() > 0 and zone_df[zone_df['is_swing']]['is_contact'].sum() / swings or 0.75,
                'swing_rate': swings / len(zone_df) if len(zone_df) > 0 else 0.5,
                'xwoba': zone_df['estimated_woba_using_speedangle'].mean() if 'estimated_woba_using_speedangle' in zone_df.columns else 0.32,
            }
        else:
            league_zone_stats[zone] = {
                'whiff_rate': 0.25,
                'contact_rate': 0.75,
                'swing_rate': 0.5,
                'xwoba': 0.32,
            }

    # Calculate stats for each batter-zone combination
    batter_zone_stats = []

    for batter_id, batter_df in df.groupby('batter'):
        batter_stats = {'batter': batter_id}

        for zone in range(1, 15):
            zone_df = batter_df[batter_df['zone'] == zone]
            n_pitches = len(zone_df)

            if n_pitches < 5:
                # Too few pitches - use league average
                for stat in ['whiff_rate', 'contact_rate', 'swing_rate', 'xwoba']:
                    batter_stats[f'zone_{zone}_{stat}'] = league_zone_stats.get(zone, {}).get(stat, 0.0)
                batter_stats[f'zone_{zone}_pitches'] = 0
                continue

            # Calculate observed stats
            swings = zone_df['is_swing'].sum()

            obs_swing_rate = swings / n_pitches if n_pitches > 0 else 0.0

            if swings > 0:
                contacts = zone_df[zone_df['is_swing']]['is_contact'].sum()
                obs_whiff_rate = 1 - (contacts / swings)
                obs_contact_rate = contacts / swings
            else:
                obs_whiff_rate = 0.0
                obs_contact_rate = 0.0

            if 'estimated_woba_using_speedangle' in zone_df.columns:
                obs_xwoba = zone_df['estimated_woba_using_speedangle'].mean()
                if pd.isna(obs_xwoba):
                    obs_xwoba = league_zone_stats.get(zone, {}).get('xwoba', 0.32)
            else:
                obs_xwoba = league_zone_stats.get(zone, {}).get('xwoba', 0.32)

            # Regression to mean based on sample size
            weight = n_pitches / (n_pitches + min_pitches)
            league = league_zone_stats.get(zone, {})

            batter_stats[f'zone_{zone}_whiff_rate'] = (
                weight * obs_whiff_rate + (1 - weight) * league.get('whiff_rate', 0.25)
            )
            batter_stats[f'zone_{zone}_contact_rate'] = (
                weight * obs_contact_rate + (1 - weight) * league.get('contact_rate', 0.75)
            )
            batter_stats[f'zone_{zone}_swing_rate'] = (
                weight * obs_swing_rate + (1 - weight) * league.get('swing_rate', 0.5)
            )
            batter_stats[f'zone_{zone}_xwoba'] = (
                weight * obs_xwoba + (1 - weight) * league.get('xwoba', 0.32)
            )
            batter_stats[f'zone_{zone}_pitches'] = n_pitches

        # Calculate aggregate chase rate (zones 11-14)
        chase_zones = batter_df[batter_df['zone'].isin([11, 12, 13, 14])]
        if len(chase_zones) > 0:
            chase_rate = chase_zones['is_swing'].sum() / len(chase_zones)
            # Regression to mean
            weight = len(chase_zones) / (len(chase_zones) + min_pitches)
            batter_stats['chase_rate_overall'] = weight * chase_rate + (1 - weight) * 0.30
        else:
            batter_stats['chase_rate_overall'] = 0.30

        batter_zone_stats.append(batter_stats)

    result_df = pd.DataFrame(batter_zone_stats)
    print(f"Built zone profiles for {len(result_df)} batters")

    return result_df


def main(input_path: Path = None, output_path: Path = None):
    """Main function to build batter zone profiles."""

    if input_path is None:
        input_path = DATA_RAW / "statcast_2024.csv"
    if output_path is None:
        output_path = DATA_PROCESSED / "batter_zone_profiles.parquet"

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    print(f"Loading data from {input_path}...")
    df = pd.read_csv(input_path, low_memory=False)
    print(f"Loaded {len(df):,} pitches")

    # Filter to pitches with zone data
    df = df[df['zone'].notna()].copy()
    print(f"Filtered to {len(df):,} pitches with zone data")

    # Build zone profiles
    zone_profiles = calculate_zone_stats(df)

    # Save
    zone_profiles.to_parquet(output_path, index=False)
    print(f"Saved zone profiles to {output_path}")

    # Summary
    print("\nZone profile summary:")
    print(f"  Batters: {len(zone_profiles)}")
    print(f"  Features per batter: {len(zone_profiles.columns) - 1}")

    # Show example
    print("\nExample batter zone stats (first batter, zones 1-5):")
    example = zone_profiles.iloc[0]
    for zone in range(1, 6):
        print(f"  Zone {zone}:")
        print(f"    Swing rate: {example[f'zone_{zone}_swing_rate']:.3f}")
        print(f"    Whiff rate: {example[f'zone_{zone}_whiff_rate']:.3f}")
        print(f"    xwOBA: {example[f'zone_{zone}_xwoba']:.3f}")

    return zone_profiles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build batter zone profiles")
    parser.add_argument("--input", type=str, help="Input Statcast CSV path")
    parser.add_argument("--output", type=str, help="Output parquet path")

    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    output_path = Path(args.output) if args.output else None

    main(input_path, output_path)
