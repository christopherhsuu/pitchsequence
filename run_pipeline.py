#!/usr/bin/env python3
"""
Master pipeline script for PitchSequence.

This script runs the entire data pipeline in order:
1. Fetch Statcast data (if not present)
2. Build batter profiles
3. Build pitcher profiles  
4. Preprocess pitch data and add run values
5. Engineer features
6. Train model

Usage:
    python run_pipeline.py                    # Run full pipeline
    python run_pipeline.py --skip-fetch       # Skip data fetch (use existing)
    python run_pipeline.py --step train       # Run only training step
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"


def run_step(script_name: str, description: str, *args):
    """Run a pipeline step."""
    print("\n" + "=" * 60)
    print(f"STEP: {description}")
    print("=" * 60)
    
    script_path = SRC / script_name
    if not script_path.exists():
        print(f"ERROR: Script not found: {script_path}")
        return False
    
    cmd = [sys.executable, str(script_path)] + list(args)
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(ROOT))
    
    if result.returncode != 0:
        print(f"ERROR: Step failed with return code {result.returncode}")
        return False
    
    print(f"✓ {description} complete")
    return True


def check_data_exists():
    """Check if raw data exists."""
    csv_files = list(DATA_RAW.glob("statcast_*.csv"))
    return len(csv_files) > 0


def main():
    parser = argparse.ArgumentParser(description="Run PitchSequence pipeline")
    parser.add_argument("--skip-fetch", action="store_true", 
                       help="Skip data fetching step")
    parser.add_argument("--step", type=str, choices=[
        "fetch", "batter", "pitcher", "preprocess", "features", "train", "all"
    ], default="all", help="Run specific step only")
    parser.add_argument("--year", type=int, default=2024,
                       help="Year to fetch data for")
    parser.add_argument("--input", type=str,
                       help="Input CSV path (for individual steps)")
    
    args = parser.parse_args()
    
    # Determine input file
    input_csv = args.input or str(DATA_RAW / f"statcast_{args.year}.csv")
    
    print("\n" + "=" * 60)
    print("PitchSequence Pipeline")
    print("=" * 60)
    print(f"Root directory: {ROOT}")
    print(f"Input data: {input_csv}")
    
    steps = {
        "fetch": ("fetch_statcast.py", "Fetching Statcast data", f"--year={args.year}"),
        "batter": ("build_batter_profiles.py", "Building batter profiles", f"--input={input_csv}"),
        "pitcher": ("build_pitcher_profiles.py", "Building pitcher profiles", f"--input={input_csv}"),
        "preprocess": ("preprocess.py", "Preprocessing pitch data", f"--input={input_csv}"),
        "features": ("feature_engineering.py", "Engineering features"),
        "train": ("train_model.py", "Training model"),
    }
    
    if args.step == "all":
        # Run all steps
        steps_to_run = ["fetch", "batter", "pitcher", "preprocess", "features", "train"]
        
        # Skip fetch if data exists and --skip-fetch
        if args.skip_fetch or check_data_exists():
            if check_data_exists():
                print(f"\n✓ Data already exists, skipping fetch")
            steps_to_run = steps_to_run[1:]  # Remove fetch
    else:
        steps_to_run = [args.step]
    
    # Run steps
    for step_name in steps_to_run:
        step_info = steps[step_name]
        script = step_info[0]
        desc = step_info[1]
        step_args = step_info[2:] if len(step_info) > 2 else []
        
        success = run_step(script, desc, *step_args)
        if not success:
            print(f"\n❌ Pipeline failed at step: {step_name}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ Pipeline complete!")
    print("=" * 60)
    
    # Print summary of outputs
    print("\nOutputs:")
    outputs = [
        (DATA_PROCESSED / "batter_profiles.parquet", "Batter profiles"),
        (DATA_PROCESSED / "batter_profiles_wide.parquet", "Batter profiles (wide)"),
        (DATA_PROCESSED / "pitcher_profiles.parquet", "Pitcher profiles"),
        (DATA_PROCESSED / "pitcher_arsenals.parquet", "Pitcher arsenals"),
        (DATA_PROCESSED / "pitches_with_rv.parquet", "Pitches with run values"),
        (DATA_PROCESSED / "features.parquet", "ML features"),
        (ARTIFACTS / "pitch_model.pkl", "Trained model"),
        (ARTIFACTS / "model_metadata.json", "Model metadata"),
    ]
    
    for path, desc in outputs:
        status = "✓" if path.exists() else "✗"
        size = f"({path.stat().st_size / 1024 / 1024:.1f} MB)" if path.exists() else ""
        print(f"  {status} {desc}: {path.name} {size}")


if __name__ == "__main__":
    main()
