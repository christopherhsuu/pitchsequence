"""
Fetch Statcast data from Baseball Savant.

Usage:
    python src/fetch_statcast.py --year 2024
    python src/fetch_statcast.py --start 2024-04-01 --end 2024-09-30
"""
import argparse
from pathlib import Path
from datetime import datetime, timedelta

try:
    from pybaseball import statcast, cache
    cache.enable()  # Cache requests to avoid re-downloading
except ImportError:
    print("Please install pybaseball: pip install pybaseball")
    exit(1)

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def fetch_statcast_data(start_dt: str, end_dt: str, output_path: Path = None) -> Path:
    """
    Fetch Statcast data for a date range.
    
    Args:
        start_dt: Start date (YYYY-MM-DD)
        end_dt: End date (YYYY-MM-DD)
        output_path: Where to save the CSV (default: data/raw/statcast_{year}.csv)
    
    Returns:
        Path to saved CSV
    """
    print(f"Fetching Statcast data from {start_dt} to {end_dt}...")
    print("This may take several minutes for large date ranges.")
    
    # Fetch data
    df = statcast(start_dt=start_dt, end_dt=end_dt)
    
    if df is None or df.empty:
        raise ValueError(f"No data returned for {start_dt} to {end_dt}")
    
    print(f"Fetched {len(df):,} pitches")
    
    # Determine output path
    if output_path is None:
        year = start_dt[:4]
        output_path = RAW_DIR / f"statcast_{year}.csv"
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    
    # Print summary
    print(f"\nData summary:")
    print(f"  Pitches: {len(df):,}")
    print(f"  Unique pitchers: {df['pitcher'].nunique():,}")
    print(f"  Unique batters: {df['batter'].nunique():,}")
    print(f"  Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print(f"  Columns: {len(df.columns)}")
    
    return output_path


def fetch_full_season(year: int) -> Path:
    """Fetch a full MLB season (April to September)."""
    start_dt = f"{year}-03-28"  # Opening day is usually late March
    end_dt = f"{year}-09-30"    # Regular season ends late September
    
    return fetch_statcast_data(start_dt, end_dt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Statcast data")
    parser.add_argument("--year", type=int, help="Fetch full season for year")
    parser.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, help="Output CSV path")
    
    args = parser.parse_args()
    
    if args.year:
        fetch_full_season(args.year)
    elif args.start and args.end:
        output = Path(args.output) if args.output else None
        fetch_statcast_data(args.start, args.end, output)
    else:
        # Default: fetch 2024 season
        print("No arguments provided. Fetching 2024 season...")
        fetch_full_season(2024)
