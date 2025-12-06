"""
Download complete Statcast data using pybaseball.

This script downloads all pitch-by-pitch data from the 2024 and 2025 seasons
without the 10,000 row limit imposed by Baseball Savant's web interface.

Usage:
    python download_statcast.py
"""
import pybaseball
from datetime import datetime
import pandas as pd
from pathlib import Path

# Enable pybaseball cache to speed up repeated downloads
pybaseball.cache.enable()

def download_season(year: int, output_dir: Path):
    """Download complete Statcast data for a season."""

    print(f"\n{'='*60}")
    print(f"Downloading {year} Statcast Data")
    print(f"{'='*60}\n")

    # Determine date range based on year
    if year == 2024:
        start_date = "2024-03-20"  # Opening day 2024
        end_date = "2024-09-30"     # End of regular season
    elif year == 2025:
        start_date = "2025-03-18"  # Estimated opening day 2025
        end_date = datetime.now().strftime("%Y-%m-%d")  # Today
    else:
        raise ValueError(f"Year {year} not configured")

    print(f"Date range: {start_date} to {end_date}")

    # Download the data
    # This may take 5-10 minutes depending on your connection
    print("Downloading... (this may take several minutes)")

    try:
        df = pybaseball.statcast(start_dt=start_date, end_dt=end_date)

        print(f"\nDownload complete!")
        print(f"Total pitches: {len(df):,}")
        print(f"Columns: {len(df.columns)}")
        print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
        print(f"Unique pitchers: {df['pitcher'].nunique()}")
        print(f"Unique batters: {df['batter'].nunique()}")

        # Save to CSV
        output_path = output_dir / f"{year}statcastdata.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved to: {output_path}")
        print(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

        return df

    except Exception as e:
        print(f"Error downloading {year} data: {e}")
        return None


def main():
    """Download both 2024 and 2025 seasons."""

    # Set output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting Statcast download...")
    print("This will download the COMPLETE dataset (no 10,000 row limit)")
    print("\nNote: This may take 10-20 minutes total for both seasons.")
    print("The pybaseball library will cache the data for future use.\n")

    # Download 2024
    df_2024 = download_season(2024, output_dir)

    # Download 2025 (if available)
    df_2025 = download_season(2025, output_dir)

    # Summary
    print(f"\n{'='*60}")
    print("Download Summary")
    print(f"{'='*60}")

    if df_2024 is not None:
        print(f"2024: {len(df_2024):,} pitches")
    if df_2025 is not None:
        print(f"2025: {len(df_2025):,} pitches")

    if df_2024 is not None and df_2025 is not None:
        total = len(df_2024) + len(df_2025)
        print(f"\nTotal: {total:,} pitches")

        # Combine for analysis
        combined = pd.concat([df_2024, df_2025], ignore_index=True)
        print(f"\nCombined dataset stats:")
        print(f"  Unique pitchers: {combined['pitcher'].nunique()}")
        print(f"  Unique batters: {combined['batter'].nunique()}")

        # Check for AJ Minter
        minter_pitches = combined[combined['pitcher'] == 621368]
        if len(minter_pitches) > 0:
            print(f"\n✓ AJ Minter (621368) found: {len(minter_pitches)} pitches")
        else:
            print(f"\n✗ AJ Minter (621368) not found in dataset")

    print(f"\n{'='*60}")
    print("Download complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
