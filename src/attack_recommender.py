import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any

import pandas as pd

# Data paths (relative to repo root)
ARCHETYPES_PATH = Path("data/player_archetypes.csv")
ARSENALS_PATH = Path("pitcher_assets/pitcher_arsenals.csv")


def load_data(archetypes_path: Path = ARCHETYPES_PATH, arsenals_path: Path = ARSENALS_PATH):
    """Load archetypes and pitcher arsenals into DataFrames.

    Returns:
        (archetypes_df, arsenals_df)
    """
    arche = pd.read_csv(archetypes_path)
    ars = pd.read_csv(arsenals_path)
    return arche, ars


def get_archetype(arche_df: pd.DataFrame, batter_id: Any) -> str:
    """Return a human-friendly archetype label for a batter id or name.

    Assumptions:
      - The archetypes CSV uses a `batter` column (player id) and contains a `cluster`
        column. If a `label` column exists it is used. Otherwise we map numeric
        clusters to sensible archetype names.
    """
    row = None
    if batter_id is None:
        return "unknown"
    # normalize input to string for comparisons
    bstr = str(batter_id).strip()
    # direct numeric id match (handles int or numeric string)
    if bstr.isdigit():
        matches = arche_df[arche_df["batter"].astype(str) == bstr]
        if not matches.empty:
            row = matches.iloc[0]
    # fallback: case-insensitive substring match against any textual columns
    if row is None:
        lowered = bstr.lower()
        def row_matches(r):
            for c in r.index:
                v = r[c]
                if pd.isna(v):
                    continue
                try:
                    if lowered in str(v).lower():
                        return True
                except Exception:
                    continue
            return False

        matches = arche_df[arche_df.apply(row_matches, axis=1)]
        if not matches.empty:
            row = matches.iloc[0]

    if row is None:
        return "unknown"

    # prefer explicit label column if present
    if "label" in arche_df.columns:
        return str(row.get("label", "unknown"))

    # map clusters (if present) to archetype names (ASSUMPTION)
    if "cluster" in arche_df.columns:
        cluster_map = {
            0: "Power Hitter",
            1: "Contact Hitter",
            2: "Contact/Puller",
            3: "Patient Hitter",
            4: "Spray/Speed"
        }
        try:
            c = int(row.get("cluster"))
            return cluster_map.get(c, f"cluster_{c}")
        except Exception:
            # if cluster is non-numeric, just return its string
            return str(row.get("cluster", "unknown"))

    # fallback: look at hard-hit/stats and derive simple label
    if "pct_hard_hit" in arche_df.columns:
        if row.get("pct_hard_hit", 0) > 0.48:
            return "Power Hitter"
    return "Unknown"


def get_pitcher_arsenal(ars_df: pd.DataFrame, pitcher_id: Any) -> List[str]:
    """Return list of pitch types for pitcher_id.

    If none found, returns a small default set.
    """
    pitches = ars_df[ars_df["pitcher"] == pitcher_id]["pitch_type"].astype(str).tolist()
    if not pitches:
        # try matching string
        pitches = ars_df[ars_df.apply(lambda r: str(pitcher_id).lower() in str(r.values).lower(), axis=1)]["pitch_type"].astype(str).tolist()
    if not pitches:
        # fallback default
        pitches = ["FF", "SL"]
    return pitches


def map_pitch_to_location(pitch_type: str, archetype: str, situation: Dict[str, Any]) -> str:
    """Return a suggested location for a pitch type given batter archetype and situation.

    This is a simple heuristic mapping.
    """
    # normalize
    p = pitch_type.upper()
    # basic locations by pitch
    loc_map = {
        "FF": "up-in",
        "FT": "up-in",
        "SI": "low-in",
        "SL": "low-away",
        "CU": "low-middle",
        "CH": "low-away",
        "CT": "arm-side edge",
        "KC": "low-away",
        "FC": "low-in",
    }
    base = loc_map.get(p, "edge-low-away")

    # archetype adjustments
    if "Power" in archetype:
        # try to keep breaking balls low-and-away
        if p in ("SL", "CU", "KC"):
            return "low-away"
        if p in ("FF", "FT"):
            return "up-in"
    if "Contact" in archetype:
        # work the edges
        if p in ("FF", "CT"):
            return "edge-away"
        return base
    if "Patient" in archetype:
        # attack early in count with strikes
        if p in ("FF", "FT"):
            return "middle-strike"
        return base

    # situation-based tweaks
    if situation.get("risp"):
        # try to induce grounders: aim low
        return base.replace("up", "low")

    return base


def recommend_sequence(arche_df: pd.DataFrame, ars_df: pd.DataFrame,
                       batter: Any, pitcher: Any, count: str,
                       situation: Dict[str, Any], seq_len: int = 3,
                       randomize: bool = True) -> Dict[str, Any]:
    """Produce a recommended pitch sequence (heuristic).

    Returns a dict with recommended_sequence (list), strategy_notes, inputs.
    """
    archetype = get_archetype(arche_df, batter)
    pitches = get_pitcher_arsenal(ars_df, pitcher)

    # Weight pitches by simple preference: breaking for power, fastball for patient, mix for contact
    def pitch_score(pt: str) -> float:
        pt = pt.upper()
        score = 1.0
        if "Power" in archetype:
            if pt in ("SL", "CU", "KC"): score += 1.5
        if "Contact" in archetype:
            if pt in ("FF", "CT", "FT"): score += 0.5
        if "Patient" in archetype:
            if pt in ("FF", "FT"): score += 1.0
        # situation tweaks
        if situation.get("late_inning") and situation.get("outs", 0) >= 2:
            # prefer chase/punchout pitches (breaking)
            if pt in ("SL", "CU"): score += 0.7
        return score

    scored = sorted(pitches, key=lambda p: pitch_score(p), reverse=True)

    sequence = []
    # Build sequence by selecting top-scored then a supporting pitch then a chase
    if not scored:
        scored = ["FF", "SL", "CU"]

    # Primary pitch
    primary = scored[0]
    sequence.append(primary)

    # Secondary: choose a pitch with different speed/shape if possible
    secondary = None
    for p in scored[1:]:
        if p != primary:
            secondary = p
            break
    if not secondary:
        secondary = primary
    sequence.append(secondary)

    # Tertiary: pick a breaking ball if not already in
    tertiary = None
    for p in scored:
        if p not in sequence and p.upper() in ("SL", "CU", "KC", "CH"):
            tertiary = p
            break
    if not tertiary:
        # fallback to any available
        tertiary = scored[min(2, len(scored)-1)]
    sequence.append(tertiary)

    # Trim/expand to seq_len
    if len(sequence) > seq_len:
        sequence = sequence[:seq_len]
    while len(sequence) < seq_len:
        sequence.append(random.choice(scored))

    # Randomize slightly to mimic variability
    if randomize:
        if random.random() < 0.15:
            random.shuffle(sequence)

    recommended = []
    for pt in sequence:
        loc = map_pitch_to_location(pt, archetype, situation)
        recommended.append({"pitch": pt, "location": loc})

    # Simple confidence heuristic
    confidence = 0.6 + 0.1 * min(3, len(pitches))
    if "Power" in archetype and any(p.upper() in ("SL", "CU", "KC") for p in pitches):
        confidence += 0.1

    result = {
        "batter": int(batter) if isinstance(batter, (int, float)) or str(batter).isdigit() else str(batter),
        "pitcher": int(pitcher) if isinstance(pitcher, (int, float)) or str(pitcher).isdigit() else str(pitcher),
        "count": count,
        "situation": situation,
        "batter_archetype": archetype,
        "recommended_sequence": recommended,
        "confidence": round(min(0.99, confidence), 2),
        "strategy_notes": generate_strategy_notes(archetype, pitches, count, situation)
    }
    return result


def get_next_pitch_candidates(arche_df: pd.DataFrame, ars_df: pd.DataFrame,
                               batter: Any, pitcher: Any, count: str,
                               situation: Dict[str, Any], top_n: int = None) -> List[Dict[str, Any]]:
    """Return ranked next-pitch candidates with percentage probabilities and suggested locations.

    Output: list of {pitch, pct, location, score}
    """
    archetype = get_archetype(arche_df, batter)
    pitches = get_pitcher_arsenal(ars_df, pitcher)

    # score function (same logic as recommend_sequence)
    def pitch_score(pt: str) -> float:
        pt = pt.upper()
        score = 1.0
        if "Power" in archetype:
            if pt in ("SL", "CU", "KC"): score += 1.5
        if "Contact" in archetype:
            if pt in ("FF", "CT", "FT"): score += 0.5
        if "Patient" in archetype:
            if pt in ("FF", "FT"): score += 1.0
        # situation tweaks
        if situation.get("late_inning") and situation.get("outs", 0) >= 2:
            if pt in ("SL", "CU"): score += 0.7
        return score

    scored = [(p, pitch_score(p)) for p in pitches]
    # ensure deterministic sort: score desc then pitch name
    scored = sorted(scored, key=lambda x: (-x[1], x[0]))

    # optionally limit
    if top_n:
        scored = scored[:top_n]

    total = sum(s for _, s in scored) if scored else 0.0
    candidates = []
    for p, s in scored:
        pct = (s / total) if total > 0 else 1.0 / max(1, len(scored))
        loc = map_pitch_to_location(p, archetype, situation)
        candidates.append({"pitch": p, "pct": round(pct * 100, 1), "location": loc, "score": round(s, 3)})

    return candidates


def generate_strategy_notes(archetype: str, pitches: List[str], count: str, situation: Dict[str, Any]) -> str:
    notes = []
    notes.append(f"Archetype detected: {archetype}.")
    notes.append(f"Pitcher arsenal: {', '.join(pitches)}.")
    if "Power" in archetype:
        notes.append("Favor breaking balls low-and-away to induce chases and weak contact.")
    if "Contact" in archetype:
        notes.append("Mix speeds and work edges to induce weak contact and avoid predictable locations.")
    if "Patient" in archetype:
        notes.append("Attack early in the count with strikes; use harder-to-hit chase pitches later.")
    if situation.get("risp"):
        notes.append("RISP: prefer low pitches to induce grounders and avoid big hits.")
    if situation.get("late_inning"):
        notes.append("Late inning: favor strikeout/weak-contact pitches.")
    notes.append(f"Count considered: {count}.")
    return " ".join(notes)


def interactive_cli():
    arche_df, ars_df = load_data()

    print("Loaded archetypes:", len(arche_df), "rows")
    print("Loaded pitcher arsenals:", len(ars_df), "rows")

    # show small sample of batters and pitchers
    sample_batters = arche_df["batter"].unique()[:40]
    sample_pitchers = ars_df["pitcher"].unique()[:40]

    print("\nSample batters (ids):", sample_batters[:20].tolist())
    print("Sample pitchers (ids):", sample_pitchers[:20].tolist())

    batter = input("Enter batter id (or partial name) from archetypes: ")
    pitcher = input("Enter pitcher id from arsenals: ")
    count = input("Enter count (e.g. 0-0, 1-2, 3-1): ") or "0-0"

    # Situation inputs
    risp = input("Runners in scoring position? (y/N): ") in ("y", "Y", "yes")
    outs = input("Outs (0/1/2): ")
    try:
        outs = int(outs)
    except Exception:
        outs = 0
    late = input("Late inning? (y/N): ") in ("y", "Y", "yes")

    situation = {"risp": risp, "outs": outs, "late_inning": late}

    res = recommend_sequence(arche_df, ars_df, batter, pitcher, count, situation)

    print(json.dumps(res, indent=2))


def example_run():
    arche_df, ars_df = load_data()
    # pick first batter and first pitcher
    batter = arche_df.iloc[0]["batter"]
    pitcher = ars_df.iloc[0]["pitcher"]
    situation = {"risp": True, "outs": 2, "late_inning": True}
    res = recommend_sequence(arche_df, ars_df, batter, pitcher, "1-1", situation)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pitch sequence recommender CLI")
    parser.add_argument("--example", action="store_true", help="Run a quick example and exit")
    args = parser.parse_args()
    if args.example:
        example_run()
    else:
        interactive_cli()
