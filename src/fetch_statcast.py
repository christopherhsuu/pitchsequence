from pybaseball import statcast
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

data = statcast(start_dt="2025-04-15", end_dt="2025-09-30")
data.to_csv(RAW_DIR / "statcast_2025.csv", index=False)
