"""
Pitch recommendation engine.

This module provides the main interface for getting pitch recommendations.
It loads the trained model and profiles, then scores all pitches in a
pitcher's arsenal for a given game state.

Usage:
    from predict import PitchRecommender
    
    recommender = PitchRecommender()
    recommendations = recommender.recommend(
        pitcher_id=543037,
        batter_id=545361,
        balls=1,
        strikes=2,
        outs=1,
        on_1b=True,
        on_2b=False,
        on_3b=False
    )
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import joblib

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import (
    DATA_PROCESSED, ARTIFACTS,
    get_pitch_category, LEAGUE_AVERAGES,
    load_run_expectancy, get_run_expectancy
)


class PitchRecommender:
    """
    Main class for pitch recommendations.
    
    Loads model and profiles on initialization, then provides
    fast recommendations for any matchup.
    """
    
    def __init__(self, model_path: Path = None, profiles_dir: Path = None):
        """
        Initialize recommender with model and profiles.
        
        Args:
            model_path: Path to trained model (default: artifacts/pitch_model.pkl)
            profiles_dir: Directory containing profiles (default: data/processed/)
        """
        if model_path is None:
            model_path = ARTIFACTS / "pitch_model.pkl"
        if profiles_dir is None:
            profiles_dir = DATA_PROCESSED
        
        self.model_path = model_path
        self.profiles_dir = profiles_dir
        
        # Load model
        if model_path.exists():
            print(f"Loading model from {model_path}")
            self.model = joblib.load(model_path)
        else:
            print(f"Warning: Model not found at {model_path}")
            self.model = None
        
        # Load inference config
        config_path = ARTIFACTS / "inference_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            self.feature_cols = config["feature_cols"]
            self.numeric_cols = config["numeric_cols"]
            self.categorical_cols = config["categorical_cols"]
        else:
            self.feature_cols = []
            self.numeric_cols = []
            self.categorical_cols = []
        
        # Load profiles
        self._load_profiles()
        
        # Load run expectancy
        self.re_matrix = load_run_expectancy()
    
    def _load_profiles(self):
        """Load batter and pitcher profiles."""
        # Batter profiles (wide format)
        batter_path = self.profiles_dir / "batter_profiles_wide.parquet"
        if batter_path.exists():
            self.batter_profiles = pd.read_parquet(batter_path)
            self.batter_profiles = self.batter_profiles.set_index("batter")
            print(f"Loaded {len(self.batter_profiles)} batter profiles")
        else:
            print(f"Warning: Batter profiles not found at {batter_path}")
            self.batter_profiles = pd.DataFrame()
        
        # Pitcher profiles
        pitcher_path = self.profiles_dir / "pitcher_profiles.parquet"
        if pitcher_path.exists():
            self.pitcher_profiles = pd.read_parquet(pitcher_path)
            print(f"Loaded {len(self.pitcher_profiles)} pitcher-pitch profiles")
        else:
            print(f"Warning: Pitcher profiles not found at {pitcher_path}")
            self.pitcher_profiles = pd.DataFrame()
        
        # Pitcher arsenals (summary)
        arsenal_path = self.profiles_dir / "pitcher_arsenals.parquet"
        if arsenal_path.exists():
            self.pitcher_arsenals = pd.read_parquet(arsenal_path)
            self.pitcher_arsenals = self.pitcher_arsenals.set_index("pitcher")
        else:
            self.pitcher_arsenals = pd.DataFrame()

        # Batter zone profiles (for location optimization)
        zone_path = self.profiles_dir / "batter_zone_profiles.parquet"
        if zone_path.exists():
            self.batter_zone_profiles = pd.read_parquet(zone_path)
            self.batter_zone_profiles = self.batter_zone_profiles.set_index("batter")
            print(f"Loaded {len(self.batter_zone_profiles)} batter zone profiles")
        else:
            print(f"Warning: Batter zone profiles not found at {zone_path}")
            self.batter_zone_profiles = pd.DataFrame()

    def _coords_to_zone(self, plate_x: float, plate_z: float) -> int:
        """
        Convert plate coordinates to zone number (1-14).

        Zones 1-9: Strike zone (3x3 grid)
        Zones 11-14: Out of zone (balls)
        """
        # Strike zone boundaries (approximate)
        sz_left = -0.83   # Left edge (from catcher's view)
        sz_right = 0.83   # Right edge
        sz_bottom = 1.5   # Bottom
        sz_top = 3.5      # Top

        # Check if in strike zone
        if sz_left <= plate_x <= sz_right and sz_bottom <= plate_z <= sz_top:
            # 3x3 grid: zones 1-9
            # Top row: 1, 2, 3 (left to right)
            # Mid row: 4, 5, 6
            # Low row: 7, 8, 9
            x_third = (sz_right - sz_left) / 3
            z_third = (sz_top - sz_bottom) / 3

            col = int((plate_x - sz_left) / x_third)  # 0, 1, 2
            row = 2 - int((plate_z - sz_bottom) / z_third)  # 2, 1, 0 (top to bottom)

            col = max(0, min(2, col))
            row = max(0, min(2, row))

            return row * 3 + col + 1
        else:
            # Out of zone: 11-14
            if plate_z > sz_top:
                return 11  # High
            elif plate_z < sz_bottom:
                return 12  # Low
            elif plate_x < sz_left:
                return 13  # Inside (to RHH)
            else:
                return 14  # Outside (to RHH)

    def _generate_location_candidates(self) -> List[Dict[str, Any]]:
        """
        Generate candidate pitch locations for grid search.

        Returns 5x5 grid covering strike zone + edges.
        """
        candidates = []

        # Horizontal: inside-inside, inside, middle, outside, outside-outside
        x_coords = [-1.2, -0.6, 0.0, 0.6, 1.2]

        # Vertical: low, mid-low, mid-mid, mid-high, high
        z_coords = [1.8, 2.3, 2.8, 3.3, 3.8]

        for x in x_coords:
            for z in z_coords:
                zone = self._coords_to_zone(x, z)
                in_zone = 1 if 1 <= zone <= 9 else 0

                candidates.append({
                    'plate_x': x,
                    'plate_z': z,
                    'zone': zone,
                    'in_zone': in_zone
                })

        return candidates

    def _get_batter_zone_stats(self, batter_id: int, zone: int) -> Dict[str, float]:
        """Get batter's performance stats for a specific zone."""
        if batter_id in self.batter_zone_profiles.index:
            profile = self.batter_zone_profiles.loc[batter_id]
            return {
                'swing_rate': profile.get(f'zone_{zone}_swing_rate', 0.5),
                'whiff_rate': profile.get(f'zone_{zone}_whiff_rate', 0.25),
                'contact_rate': profile.get(f'zone_{zone}_contact_rate', 0.75),
                'xwoba': profile.get(f'zone_{zone}_xwoba', 0.32),
            }
        else:
            # Return league averages
            return {
                'swing_rate': 0.5,
                'whiff_rate': 0.25,
                'contact_rate': 0.75,
                'xwoba': 0.32,
            }

    def optimize_pitch_location(
        self,
        pitcher_id: int,
        batter_id: int,
        pitch_type: str,
        balls: int = 0,
        strikes: int = 0,
        outs: int = 0,
        on_1b: bool = False,
        on_2b: bool = False,
        on_3b: bool = False,
        stand: str = "R",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Find optimal location for a pitch using grid search.

        Evaluates all candidate locations and returns the one with
        the lowest predicted run value.

        Returns:
            Dict with plate_x, plate_z, zone, in_zone, expected_rv
        """
        if self.model is None:
            # Fallback: return middle-low location
            return {
                'plate_x': 0.0,
                'plate_z': 2.3,
                'zone': 8,
                'in_zone': 1,
                'expected_rv': 0.0
            }

        candidates = self._generate_location_candidates()

        best_location = None
        best_rv = float('inf')

        for loc in candidates:
            # Build feature row with this location
            features = self.build_feature_row(
                pitcher_id=pitcher_id,
                batter_id=batter_id,
                pitch_type=pitch_type,
                balls=balls,
                strikes=strikes,
                outs=outs,
                on_1b=on_1b,
                on_2b=on_2b,
                on_3b=on_3b,
                stand=stand,
                **kwargs
            )

            # Add location features
            features['plate_x'] = loc['plate_x']
            features['plate_z'] = loc['plate_z']
            features['zone'] = loc['zone']
            features['in_zone'] = loc['in_zone']

            # Add discretized location categories
            if loc['plate_x'] < -1.0:
                features['loc_horizontal'] = 'inside_inside'
            elif loc['plate_x'] < -0.3:
                features['loc_horizontal'] = 'inside'
            elif loc['plate_x'] < 0.3:
                features['loc_horizontal'] = 'middle'
            elif loc['plate_x'] < 1.0:
                features['loc_horizontal'] = 'outside'
            else:
                features['loc_horizontal'] = 'outside_outside'

            if loc['plate_z'] < 1.8:
                features['loc_vertical'] = 'low'
            elif loc['plate_z'] < 2.5:
                features['loc_vertical'] = 'mid_low'
            elif loc['plate_z'] < 3.5:
                features['loc_vertical'] = 'mid_high'
            else:
                features['loc_vertical'] = 'high'

            # Get batter zone stats for this location
            zone_stats = self._get_batter_zone_stats(batter_id, int(loc['zone']))
            features['batter_swing_rate_at_zone'] = zone_stats['swing_rate']
            features['batter_whiff_rate_at_zone'] = zone_stats['whiff_rate']
            features['batter_contact_rate_at_zone'] = zone_stats['contact_rate']
            features['batter_xwoba_at_zone'] = zone_stats['xwoba']

            # Convert to DataFrame and predict
            df = pd.DataFrame([features])
            X = df[[c for c in self.feature_cols if c in df.columns]].copy()

            # Add missing columns with defaults
            for col in self.feature_cols:
                if col not in X.columns:
                    X[col] = 0

            X = X[self.feature_cols]  # Ensure correct order

            predicted_rv = self.model.predict(X)[0]

            if predicted_rv < best_rv:
                best_rv = predicted_rv
                best_location = loc.copy()

        best_location['expected_rv'] = best_rv
        return best_location

    def get_pitcher_arsenal(self, pitcher_id: int) -> List[str]:
        """Get list of pitch types for a pitcher."""
        if pitcher_id in self.pitcher_arsenals.index:
            pitch_str = self.pitcher_arsenals.loc[pitcher_id, "pitch_types"]
            return pitch_str.split(",") if isinstance(pitch_str, str) else []
        
        # Fallback: check pitcher_profiles
        if len(self.pitcher_profiles) > 0:
            pitcher_data = self.pitcher_profiles[
                self.pitcher_profiles["pitcher"] == pitcher_id
            ]
            if len(pitcher_data) > 0:
                return pitcher_data["pitch_type"].unique().tolist()
        
        # Default arsenal
        return ["FF", "SL", "CH"]
    
    def get_batter_profile(self, batter_id: int) -> Dict[str, float]:
        """Get batter vulnerability profile."""
        if batter_id in self.batter_profiles.index:
            return self.batter_profiles.loc[batter_id].to_dict()
        
        # Return league averages if batter not found
        profile = {}
        for cat in ["FB", "BR", "OS"]:
            for stat in ["whiff_rate", "chase_rate", "zone_contact_rate", "gb_rate", "xwoba"]:
                profile[f"{stat}_vs_{cat}"] = LEAGUE_AVERAGES[cat][stat]
        return profile
    
    def get_pitcher_pitch_profile(self, pitcher_id: int, pitch_type: str) -> Dict[str, float]:
        """Get pitcher's characteristics for a specific pitch type."""
        if len(self.pitcher_profiles) > 0:
            mask = (
                (self.pitcher_profiles["pitcher"] == pitcher_id) &
                (self.pitcher_profiles["pitch_type"] == pitch_type)
            )
            data = self.pitcher_profiles[mask]
            if len(data) > 0:
                row = data.iloc[0]
                return {
                    "pitcher_avg_velo": row.get("pitcher_avg_velo", 93.0),
                    "pitcher_avg_spin": row.get("pitcher_avg_spin", 2200),
                    "pitcher_avg_pfx_x": row.get("pitcher_avg_pfx_x", 0),
                    "pitcher_avg_pfx_z": row.get("pitcher_avg_pfx_z", 0),
                    "pitcher_whiff_rate": row.get("pitcher_whiff_rate", 0.25),
                    "pitcher_strike_rate": row.get("pitcher_strike_rate", 0.65),
                    "pitcher_usage_rate": row.get("pitcher_usage_rate", 0.2),
                    "pitcher_velo_diff_from_fb": row.get("pitcher_velo_diff_from_fb", 0),
                    "p_throws": row.get("p_throws", "R"),
                }
        
        # Defaults by pitch category
        cat = get_pitch_category(pitch_type)
        defaults = {
            "FB": {"pitcher_avg_velo": 94, "pitcher_avg_spin": 2300, "pitcher_velo_diff_from_fb": 0},
            "BR": {"pitcher_avg_velo": 84, "pitcher_avg_spin": 2600, "pitcher_velo_diff_from_fb": 10},
            "OS": {"pitcher_avg_velo": 86, "pitcher_avg_spin": 1800, "pitcher_velo_diff_from_fb": 8},
        }
        base = defaults.get(cat, defaults["FB"])
        return {
            **base,
            "pitcher_avg_pfx_x": 0,
            "pitcher_avg_pfx_z": 0,
            "pitcher_whiff_rate": 0.25,
            "pitcher_strike_rate": 0.65,
            "pitcher_usage_rate": 0.2,
            "p_throws": "R",
        }
    
    def build_feature_row(
        self,
        pitcher_id: int,
        batter_id: int,
        pitch_type: str,
        balls: int = 0,
        strikes: int = 0,
        outs: int = 0,
        on_1b: bool = False,
        on_2b: bool = False,
        on_3b: bool = False,
        stand: str = "R",
        last_pitch_type: str = None,
        pitch_num_in_ab: int = 1,
        inning: int = 1,
        score_diff: int = 0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Build a feature row for model prediction.
        
        This creates all the features the model expects for a single
        pitch type option.
        """
        # Get profiles
        batter_profile = self.get_batter_profile(batter_id)
        pitcher_profile = self.get_pitcher_pitch_profile(pitcher_id, pitch_type)
        
        pitch_category = get_pitch_category(pitch_type)
        last_pitch_category = get_pitch_category(last_pitch_type) if last_pitch_type else None
        
        # Build feature dict
        features = {
            # Count
            "count_balls": balls,
            "count_strikes": strikes,
            "is_ahead_in_count": int(strikes > balls),
            "is_behind_in_count": int(balls > strikes),
            "is_two_strikes": int(strikes == 2),
            "is_three_balls": int(balls == 3),
            "is_full_count": int(balls == 3 and strikes == 2),
            "is_first_pitch": int(balls == 0 and strikes == 0),
            
            # Situation
            "outs": outs,
            "is_two_outs": int(outs == 2),
            "on_1b": int(on_1b),
            "on_2b": int(on_2b),
            "on_3b": int(on_3b),
            "runners_on": int(on_1b) + int(on_2b) + int(on_3b),
            "risp": int(on_2b or on_3b),
            "bases_empty": int(not (on_1b or on_2b or on_3b)),
            "bases_loaded": int(on_1b and on_2b and on_3b),
            "inning_num": inning,
            "is_late_inning": int(inning >= 7),
            "score_diff": score_diff,
            "is_close_game": int(abs(score_diff) <= 2),
            
            # Matchup
            "stand": stand,
            "p_throws": pitcher_profile.get("p_throws", "R"),
            "same_hand": int(stand == pitcher_profile.get("p_throws", "R")),
            "platoon_advantage": int(stand != pitcher_profile.get("p_throws", "R")),
            
            # Pitch info
            "pitch_type": pitch_type,
            "pitch_category": pitch_category,
            
            # Sequence
            "last_pitch_type": last_pitch_type,
            "last_pitch_category": last_pitch_category,
            "pitch_num_in_ab": pitch_num_in_ab,
            "same_as_last": int(pitch_type == last_pitch_type) if last_pitch_type else 0,
            "consecutive_same_pitch": 0,  # Would need history
            "speed_delta": 0,  # Would need last pitch speed
            
            # Batter vulnerability for this pitch category
            "batter_whiff_rate_vs_this_cat": batter_profile.get(f"whiff_rate_vs_{pitch_category}", 0.25),
            "batter_chase_rate_vs_this_cat": batter_profile.get(f"chase_rate_vs_{pitch_category}", 0.30),
            "batter_zone_contact_rate_vs_this_cat": batter_profile.get(f"zone_contact_rate_vs_{pitch_category}", 0.85),
            "batter_gb_rate_vs_this_cat": batter_profile.get(f"gb_rate_vs_{pitch_category}", 0.45),
            "batter_xwoba_vs_this_cat": batter_profile.get(f"xwoba_vs_{pitch_category}", 0.320),
            
            # All batter splits
            **{f"whiff_rate_vs_{cat}": batter_profile.get(f"whiff_rate_vs_{cat}", LEAGUE_AVERAGES[cat]["whiff_rate"]) 
               for cat in ["FB", "BR", "OS"]},
            **{f"chase_rate_vs_{cat}": batter_profile.get(f"chase_rate_vs_{cat}", LEAGUE_AVERAGES[cat]["chase_rate"]) 
               for cat in ["FB", "BR", "OS"]},
            **{f"xwoba_vs_{cat}": batter_profile.get(f"xwoba_vs_{cat}", LEAGUE_AVERAGES[cat]["xwoba"]) 
               for cat in ["FB", "BR", "OS"]},
            
            # Pitcher characteristics
            **pitcher_profile,
            
            # Interactions
            "whiff_interaction": (
                batter_profile.get(f"whiff_rate_vs_{pitch_category}", 0.25) *
                pitcher_profile.get("pitcher_whiff_rate", 0.25)
            ),
            "chase_opportunity": (
                batter_profile.get(f"chase_rate_vs_{pitch_category}", 0.30) *
                int(strikes > balls)
            ),
            "speed_change_magnitude": abs(pitcher_profile.get("pitcher_velo_diff_from_fb", 0)),
            "two_strike_whiff_boost": (
                int(strikes == 2) *
                batter_profile.get(f"whiff_rate_vs_{pitch_category}", 0.25)
            ),
        }
        
        return features
    
    def recommend(
        self,
        pitcher_id: int,
        batter_id: int,
        balls: int = 0,
        strikes: int = 0,
        outs: int = 0,
        on_1b: bool = False,
        on_2b: bool = False,
        on_3b: bool = False,
        stand: str = "R",
        last_pitch_type: str = None,
        pitch_num_in_ab: int = 1,
        inning: int = 1,
        score_diff: int = 0,
        top_n: int = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Get pitch recommendations for a matchup.
        
        Returns list of dicts with:
        - pitch_type: The pitch
        - predicted_rv: Model's predicted run value (lower = better for pitcher)
        - probability: Softmax probability (higher = recommended)
        - category: FB/BR/OS
        - reasoning: Why this pitch might work
        
        Sorted by probability (best first).
        """
        # Get pitcher's arsenal
        arsenal = self.get_pitcher_arsenal(pitcher_id)
        
        if not arsenal:
            return [{"error": "No arsenal found for pitcher"}]
        
        # Build feature rows for each pitch option
        rows = []
        for pitch_type in arsenal:
            features = self.build_feature_row(
                pitcher_id=pitcher_id,
                batter_id=batter_id,
                pitch_type=pitch_type,
                balls=balls,
                strikes=strikes,
                outs=outs,
                on_1b=on_1b,
                on_2b=on_2b,
                on_3b=on_3b,
                stand=stand,
                last_pitch_type=last_pitch_type,
                pitch_num_in_ab=pitch_num_in_ab,
                inning=inning,
                score_diff=score_diff,
                **kwargs
            )
            rows.append(features)
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Get predictions
        if self.model is not None:
            # Filter to feature columns the model expects
            X = df[[c for c in self.feature_cols if c in df.columns]].copy()

            # Add missing columns with defaults
            for col in self.feature_cols:
                if col not in X.columns:
                    X[col] = 0

            X = X[self.feature_cols]  # Ensure correct order
            
            predicted_rv = self.model.predict(X)
        else:
            # Fallback: use heuristic based on batter vulnerability
            predicted_rv = []
            for i, row in df.iterrows():
                # Lower vulnerability = higher RV (worse for pitcher)
                vuln = row.get("batter_whiff_rate_vs_this_cat", 0.25)
                rv = 0.1 - vuln * 0.4  # Simple heuristic
                predicted_rv.append(rv)
            predicted_rv = np.array(predicted_rv)
        
        # Convert to probabilities (softmax on negative RV)
        # Lower RV = better for pitcher = higher probability
        neg_rv = -predicted_rv
        exp_rv = np.exp(neg_rv - neg_rv.max())  # Numerical stability
        probabilities = exp_rv / exp_rv.sum()
        
        # Get current run expectancy for context
        current_re = get_run_expectancy(outs, on_1b, on_2b, on_3b, self.re_matrix)
        
        # Build results with location optimization
        results = []
        for i, pitch_type in enumerate(arsenal):
            cat = get_pitch_category(pitch_type)
            batter_whiff = df.iloc[i].get("batter_whiff_rate_vs_this_cat", 0.25)
            batter_chase = df.iloc[i].get("batter_chase_rate_vs_this_cat", 0.30)
            batter_xwoba = df.iloc[i].get("batter_xwoba_vs_this_cat", 0.320)

            # Optimize location for this pitch type
            optimal_location = self.optimize_pitch_location(
                pitcher_id=pitcher_id,
                batter_id=batter_id,
                pitch_type=pitch_type,
                balls=balls,
                strikes=strikes,
                outs=outs,
                on_1b=on_1b,
                on_2b=on_2b,
                on_3b=on_3b,
                stand=stand,
                last_pitch_type=last_pitch_type,
                pitch_num_in_ab=pitch_num_in_ab,
                inning=inning,
                score_diff=score_diff,
                **kwargs
            )

            # Generate reasoning
            reasoning = self._generate_reasoning(
                pitch_type, cat, batter_whiff, batter_chase, batter_xwoba,
                balls, strikes, optimal_location['expected_rv']
            )

            results.append({
                "pitch_type": pitch_type,
                "category": cat,
                "predicted_rv": float(predicted_rv[i]),
                "expected_re_after": float(current_re + predicted_rv[i]),
                "probability": float(probabilities[i]),
                "probability_pct": float(probabilities[i] * 100),
                "batter_whiff_rate": float(batter_whiff),
                "batter_chase_rate": float(batter_chase),
                "batter_xwoba": float(batter_xwoba),
                "reasoning": reasoning,
                # Location recommendation
                "location": {
                    "plate_x": float(optimal_location['plate_x']),
                    "plate_z": float(optimal_location['plate_z']),
                    "zone": int(optimal_location['zone']),
                    "in_zone": bool(optimal_location['in_zone']),
                    "expected_rv": float(optimal_location['expected_rv'])
                }
            })
        
        # Sort by probability (highest first)
        results = sorted(results, key=lambda x: -x["probability"])
        
        if top_n:
            results = results[:top_n]
        
        return results
    
    def _generate_reasoning(
        self,
        pitch_type: str,
        category: str,
        whiff_rate: float,
        chase_rate: float,
        xwoba: float,
        balls: int,
        strikes: int,
        predicted_rv: float
    ) -> str:
        """Generate human-readable reasoning for a recommendation."""
        reasons = []
        
        # Whiff rate analysis
        league_whiff = LEAGUE_AVERAGES[category]["whiff_rate"]
        if whiff_rate > league_whiff + 0.05:
            reasons.append(f"Batter has high whiff rate vs {category} ({whiff_rate:.0%})")
        elif whiff_rate < league_whiff - 0.05:
            reasons.append(f"Batter makes good contact vs {category}")
        
        # Chase rate analysis
        league_chase = LEAGUE_AVERAGES[category]["chase_rate"]
        if chase_rate > league_chase + 0.05:
            reasons.append(f"Batter chases {category} pitches ({chase_rate:.0%})")
        
        # xwOBA analysis
        if xwoba < 0.280:
            reasons.append(f"Weak contact expected ({xwoba:.3f} xwOBA)")
        elif xwoba > 0.360:
            reasons.append(f"Dangerous contact potential ({xwoba:.3f} xwOBA)")
        
        # Count context
        if strikes == 2:
            if category in ["BR", "OS"]:
                reasons.append("Put-away count favors offspeed/breaking")
        if balls == 3:
            if category == "FB":
                reasons.append("Need strike - fastball for command")
        
        return "; ".join(reasons) if reasons else "Standard option"


# Convenience function for quick recommendations
def recommend_pitch(
    pitcher_id: int,
    batter_id: int,
    **kwargs
) -> List[Dict[str, Any]]:
    """
    Quick function to get pitch recommendations.
    
    Creates a recommender instance and returns recommendations.
    """
    recommender = PitchRecommender()
    return recommender.recommend(pitcher_id, batter_id, **kwargs)


# For backwards compatibility with old API
def recommend_next_pitch(state: dict, pitcher_id: int, batter_stats_map: dict = None):
    """
    Legacy API wrapper.
    
    Returns: (best_pitch, best_rv, all_predictions, debug_info)
    """
    recommender = PitchRecommender()
    
    recommendations = recommender.recommend(
        pitcher_id=pitcher_id,
        batter_id=state.get("batter_id", 0),
        balls=state.get("balls", 0),
        strikes=state.get("strikes", 0),
        outs=state.get("outs_when_up", 0),
        on_1b=bool(state.get("on_1b")),
        on_2b=bool(state.get("on_2b")),
        on_3b=bool(state.get("on_3b")),
        stand=state.get("stand", "R"),
        last_pitch_type=state.get("last_pitch_type"),
    )
    
    if not recommendations:
        return "FF", 0.0, {}, []
    
    best = recommendations[0]
    preds_dict = {r["pitch_type"]: r["predicted_rv"] for r in recommendations}
    
    return best["pitch_type"], best["predicted_rv"], preds_dict, recommendations


if __name__ == "__main__":
    # Test the recommender
    print("Testing PitchRecommender...")
    
    recommender = PitchRecommender()
    
    # Example recommendation
    results = recommender.recommend(
        pitcher_id=543037,  # Example pitcher
        batter_id=545361,   # Example batter
        balls=1,
        strikes=2,
        outs=1,
        on_1b=True,
        on_2b=False,
        on_3b=False,
        stand="R"
    )
    
    print("\nRecommendations:")
    for i, r in enumerate(results):
        print(f"\n{i+1}. {r['pitch_type']} ({r['category']})")
        print(f"   Probability: {r['probability_pct']:.1f}%")
        print(f"   Predicted RV: {r['predicted_rv']:.4f}")
        print(f"   Batter whiff rate: {r['batter_whiff_rate']:.1%}")
        print(f"   Reasoning: {r['reasoning']}")
