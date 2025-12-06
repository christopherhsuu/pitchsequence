"""
Shared utility functions for the PitchSequence pipeline.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
STATIC = ROOT / "static"
ARTIFACTS = ROOT / "artifacts"

# Ensure directories exist
for d in [DATA_RAW, DATA_PROCESSED, STATIC, ARTIFACTS]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Pitch Category Mapping
# =============================================================================

PITCH_CATEGORIES = {
    # Fastballs
    "FF": "FB",  # 4-seam fastball
    "SI": "FB",  # Sinker
    "FC": "FB",  # Cutter
    "FT": "FB",  # 2-seam fastball
    "FA": "FB",  # Fastball (generic)
    
    # Breaking balls
    "SL": "BR",  # Slider
    "CU": "BR",  # Curveball
    "KC": "BR",  # Knuckle curve
    "ST": "BR",  # Sweeper
    "SV": "BR",  # Slurve
    "CS": "BR",  # Slow curve
    
    # Offspeed
    "CH": "OS",  # Changeup
    "FS": "OS",  # Splitter
    "SC": "OS",  # Screwball
    "KN": "OS",  # Knuckleball
    "EP": "OS",  # Eephus
    
    # Other (map to closest)
    "PO": "OS",  # Pitch out
    "IN": "FB",  # Intentional ball
}

def get_pitch_category(pitch_type: str) -> str:
    """Map pitch type to category (FB/BR/OS)."""
    if pd.isna(pitch_type):
        return "UNKNOWN"
    return PITCH_CATEGORIES.get(str(pitch_type).upper(), "UNKNOWN")


# =============================================================================
# Run Expectancy Matrix
# =============================================================================

def load_run_expectancy(path: Optional[Path] = None) -> Dict[Tuple[int, str], float]:
    """
    Load run expectancy matrix.
    
    Returns dict: (outs, bases_state) -> expected_runs
    bases_state is string like "000", "100", "110", etc.
    """
    if path is None:
        path = STATIC / "run_expectancy.csv"
    
    if not path.exists():
        # Use 2024 league averages if file doesn't exist
        return _default_run_expectancy()
    
    df = pd.read_csv(path, dtype={"bases_state": str})
    return {
        (int(row.outs), str(row.bases_state).zfill(3)): float(row.run_expectancy)
        for row in df.itertuples()
    }


def _default_run_expectancy() -> Dict[Tuple[int, str], float]:
    """Default RE24 matrix based on 2024 MLB averages."""
    return {
        # (outs, bases_state) -> expected_runs
        # Format: bases_state = "1B_2B_3B" where 1=occupied, 0=empty
        (0, "000"): 0.481,
        (0, "100"): 0.859,
        (0, "010"): 1.100,
        (0, "001"): 1.392,
        (0, "110"): 1.437,
        (0, "101"): 1.790,
        (0, "011"): 1.966,
        (0, "111"): 2.282,
        (1, "000"): 0.254,
        (1, "100"): 0.509,
        (1, "010"): 0.664,
        (1, "001"): 0.950,
        (1, "110"): 0.884,
        (1, "101"): 1.130,
        (1, "011"): 1.376,
        (1, "111"): 1.520,
        (2, "000"): 0.098,
        (2, "100"): 0.224,
        (2, "010"): 0.319,
        (2, "001"): 0.363,
        (2, "110"): 0.429,
        (2, "101"): 0.478,
        (2, "011"): 0.580,
        (2, "111"): 0.736,
    }


def bases_to_state(on_1b, on_2b, on_3b) -> str:
    """Convert base runner booleans to state string."""
    return f"{int(bool(on_1b))}{int(bool(on_2b))}{int(bool(on_3b))}"


def get_run_expectancy(outs: int, on_1b, on_2b, on_3b, re_matrix: Dict = None) -> float:
    """Get run expectancy for a game state."""
    if re_matrix is None:
        re_matrix = load_run_expectancy()
    
    state = bases_to_state(on_1b, on_2b, on_3b)
    outs = min(2, max(0, int(outs)))  # Clamp to 0-2
    
    return re_matrix.get((outs, state), 0.0)


# =============================================================================
# Count-Based Run Value Deltas
# =============================================================================

# Run value change for count transitions (pitcher perspective, negative = good)
# These are based on linear weights research
COUNT_DELTAS = {
    # (balls, strikes) -> delta for adding a ball
    "ball": {
        (0, 0): 0.032,
        (1, 0): 0.039,
        (2, 0): 0.059,
        (3, 0): 0.0,  # Walk handled separately
        (0, 1): 0.025,
        (1, 1): 0.032,
        (2, 1): 0.048,
        (3, 1): 0.0,
        (0, 2): 0.021,
        (1, 2): 0.027,
        (2, 2): 0.037,
        (3, 2): 0.0,
    },
    # (balls, strikes) -> delta for adding a strike (called or swinging)
    "strike": {
        (0, 0): -0.038,
        (1, 0): -0.043,
        (2, 0): -0.047,
        (3, 0): -0.051,
        (0, 1): -0.049,
        (1, 1): -0.056,
        (2, 1): -0.066,
        (3, 1): -0.078,
        (0, 2): 0.0,  # Strikeout handled separately
        (1, 2): 0.0,
        (2, 2): 0.0,
        (3, 2): 0.0,
    },
}


def get_count_delta(balls: int, strikes: int, outcome: str) -> float:
    """
    Get run value change for a count transition.
    
    outcome: 'ball', 'strike', 'foul'
    Returns positive value = bad for pitcher, negative = good for pitcher
    """
    balls = min(3, max(0, int(balls)))
    strikes = min(2, max(0, int(strikes)))
    
    if outcome == "ball":
        return COUNT_DELTAS["ball"].get((balls, strikes), 0.03)
    elif outcome in ("strike", "swinging_strike", "called_strike"):
        return COUNT_DELTAS["strike"].get((balls, strikes), -0.04)
    elif outcome == "foul":
        if strikes < 2:
            return COUNT_DELTAS["strike"].get((balls, strikes), -0.04)
        return 0.0  # Foul with 2 strikes doesn't change count
    
    return 0.0


# =============================================================================
# Regression to Mean (Bayesian shrinkage)
# =============================================================================

def regress_to_mean(
    observed_value: float,
    sample_size: int,
    population_mean: float,
    population_variance: float = 0.01,
    reliability_threshold: int = 100
) -> float:
    """
    Apply regression to the mean for small samples.
    
    Uses a simple shrinkage formula:
    weight = n / (n + k)
    where k is chosen so that at reliability_threshold samples, we trust the observed 80%
    
    Args:
        observed_value: The raw observed statistic
        sample_size: Number of observations
        population_mean: League/population average for this stat
        population_variance: How much the stat varies in the population
        reliability_threshold: Sample size at which to trust observed ~80%
    
    Returns:
        Shrunk estimate
    """
    if sample_size <= 0:
        return population_mean
    
    # Calculate shrinkage weight
    # This gives weight ~0.5 at n=reliability_threshold/2, ~0.8 at n=reliability_threshold
    k = reliability_threshold / 4
    weight = sample_size / (sample_size + k)
    
    return weight * observed_value + (1 - weight) * population_mean


# =============================================================================
# Feature Aggregation Helpers
# =============================================================================

def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default for zero denominator."""
    if denominator == 0 or pd.isna(denominator):
        return default
    return numerator / denominator


def calculate_whiff_rate(swings: int, whiffs: int) -> float:
    """Calculate whiff rate (swinging strikes / swings)."""
    return safe_divide(whiffs, swings, default=0.0)


def calculate_chase_rate(out_of_zone: int, chases: int) -> float:
    """Calculate chase rate (swings at balls / pitches out of zone)."""
    return safe_divide(chases, out_of_zone, default=0.0)


def calculate_zone_contact_rate(in_zone_swings: int, in_zone_contact: int) -> float:
    """Calculate zone contact rate (contact / swings in zone)."""
    return safe_divide(in_zone_contact, in_zone_swings, default=0.0)


# =============================================================================
# League Average Stats (2024 baselines)
# =============================================================================

LEAGUE_AVERAGES = {
    "FB": {
        "whiff_rate": 0.22,
        "chase_rate": 0.27,
        "zone_contact_rate": 0.87,
        "xwoba": 0.340,
        "gb_rate": 0.40,
    },
    "BR": {
        "whiff_rate": 0.35,
        "chase_rate": 0.32,
        "zone_contact_rate": 0.78,
        "xwoba": 0.280,
        "gb_rate": 0.45,
    },
    "OS": {
        "whiff_rate": 0.32,
        "chase_rate": 0.30,
        "zone_contact_rate": 0.82,
        "xwoba": 0.300,
        "gb_rate": 0.48,
    },
}


def get_league_average(pitch_category: str, stat: str) -> float:
    """Get league average for a stat by pitch category."""
    cat_stats = LEAGUE_AVERAGES.get(pitch_category, LEAGUE_AVERAGES["FB"])
    return cat_stats.get(stat, 0.0)


# =============================================================================
# Outcome Classification
# =============================================================================

def classify_outcome(description: str, events: str = None) -> str:
    """
    Classify pitch outcome into categories.
    
    Returns one of:
    - 'ball'
    - 'called_strike'
    - 'swinging_strike'
    - 'foul'
    - 'in_play_out'
    - 'in_play_hit'
    - 'other'
    """
    if pd.isna(description):
        return "other"
    
    desc = str(description).lower()
    
    if "ball" in desc and "foul" not in desc:
        return "ball"
    if "called_strike" in desc:
        return "called_strike"
    if "swinging_strike" in desc or "missed_bunt" in desc:
        return "swinging_strike"
    if "foul" in desc:
        return "foul"
    if "in_play" in desc or "hit_into" in desc:
        # Check events to determine if out or hit
        if events and not pd.isna(events):
            events_lower = str(events).lower()
            if any(h in events_lower for h in ["single", "double", "triple", "home_run"]):
                return "in_play_hit"
        return "in_play_out"
    
    return "other"


def is_swing(description: str) -> bool:
    """Check if the batter swung at the pitch."""
    if pd.isna(description):
        return False
    desc = str(description).lower()
    return any(s in desc for s in ["swing", "foul", "in_play", "hit_into", "missed_bunt"])


def is_in_zone(zone: int) -> bool:
    """Check if pitch was in the strike zone (zones 1-9)."""
    if pd.isna(zone):
        return False
    return 1 <= int(zone) <= 9


def is_contact(description: str) -> bool:
    """Check if batter made contact."""
    if pd.isna(description):
        return False
    desc = str(description).lower()
    return any(c in desc for c in ["foul", "in_play", "hit_into"])


# =============================================================================
# Statcast Column Names
# =============================================================================

# Key columns we need from Statcast
STATCAST_COLUMNS = [
    # Identifiers
    "game_pk", "game_date", "pitcher", "batter", "at_bat_number", "pitch_number",
    
    # Count and situation
    "balls", "strikes", "outs_when_up", "inning", "inning_topbot",
    "on_1b", "on_2b", "on_3b",
    
    # Pitch info
    "pitch_type", "release_speed", "release_spin_rate",
    "pfx_x", "pfx_z",  # Movement
    "plate_x", "plate_z",  # Location
    "zone",
    
    # Player info
    "stand", "p_throws",
    
    # Outcomes
    "description", "events", "type",
    "launch_speed", "launch_angle",
    "estimated_woba_using_speedangle",  # xwOBA
    "woba_value",
    
    # Game context
    "home_score", "away_score", "bat_score", "fld_score",
]


def load_statcast(path: Path) -> pd.DataFrame:
    """Load and validate Statcast data."""
    df = pd.read_csv(path, low_memory=False)
    
    # Find available columns
    available = [c for c in STATCAST_COLUMNS if c in df.columns]
    missing = [c for c in STATCAST_COLUMNS if c not in df.columns]
    
    if missing:
        print(f"Note: Missing columns: {missing}")
    
    return df[available].copy()
