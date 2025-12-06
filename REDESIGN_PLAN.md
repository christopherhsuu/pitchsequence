# PitchSequence Redesign Plan

## Goal
Build a pitch recommendation system that suggests the optimal next pitch to minimize expected run value, accounting for:
1. Batter-specific vulnerabilities (not just archetypes)
2. Pitch sequencing effects
3. Game situation context
4. Real-time usability

## Core Philosophy

### Why the Old Approach Falls Short

The previous clustering approach grouped batters by *batting style* (power hitter, contact hitter, etc.), but this doesn't directly map to *pitch vulnerabilities*. A "power hitter" label tells you how they hit, not what they can't hit.

### The New Approach

We'll predict pitch outcomes directly using batter-specific features that capture vulnerabilities:

```
Input: [game_state] + [pitch_type] + [batter_vulnerability_features] + [sequence_features]
Output: Expected run value for this pitch
```

The model learns: "Given this batter's chase rate vs breaking balls is 35% and their whiff rate vs sliders is 28%, throwing a slider in this count yields an expected run value of -0.03"

## Data Pipeline

```
statcast_raw.csv
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: fetch_statcast.py                                           │
│ - Download raw pitch-by-pitch data from Baseball Savant             │
│ - Output: data/raw/statcast_YYYY.csv                                │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: build_batter_profiles.py                                    │
│ - Aggregate batter stats by pitch category (FB/BR/OS)               │
│ - Calculate: whiff_rate, chase_rate, contact_rate, xwOBA            │
│ - Apply regression to mean for small samples                        │
│ - Output: data/processed/batter_profiles.parquet                    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: build_pitcher_profiles.py                                   │
│ - Build pitcher arsenals with pitch characteristics                 │
│ - Calculate: avg velocity, movement, spin by pitch type             │
│ - Output: data/processed/pitcher_profiles.parquet                   │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: preprocess.py                                               │
│ - Clean raw pitch data                                              │
│ - Calculate run values using RE24 matrix                            │
│ - Output: data/processed/pitches_with_rv.parquet                    │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: feature_engineering.py                                      │
│ - Join batter/pitcher profiles to pitch data                        │
│ - Add sequence features (last pitch, speed delta, etc.)             │
│ - Add situational features                                          │
│ - Output: data/processed/features.parquet                           │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: train_model.py                                              │
│ - Train gradient boosting model to predict run value                │
│ - Evaluate with time-based cross-validation                         │
│ - Output: artifacts/pitch_model.pkl                                 │
└─────────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────────┐
│ Step 7: predict.py                                                  │
│ - Load model and profiles                                           │
│ - Score all pitches in pitcher's arsenal                            │
│ - Return ranked recommendations                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Feature Design

### Batter Vulnerability Features (Per Pitch Category)

We aggregate stats into 3 pitch categories to ensure sufficient sample size:
- **FB** (Fastball): FF, SI, FC, FT
- **BR** (Breaking): SL, CU, KC, ST, SV
- **OS** (Offspeed): CH, FS, SC

For each category, we calculate:

| Feature | Description | Why It Matters |
|---------|-------------|----------------|
| `whiff_rate_vs_{cat}` | Swings & misses / swings | Direct measure of deception |
| `chase_rate_vs_{cat}` | Swings at balls / balls seen | Will they expand zone? |
| `zone_contact_rate_vs_{cat}` | Contact on pitches in zone | Can they catch up? |
| `xwoba_vs_{cat}` | Expected wOBA when contact made | Quality of contact |
| `gb_rate_vs_{cat}` | Ground balls / balls in play | Batted ball tendency |

### Sequence Features

| Feature | Description |
|---------|-------------|
| `last_pitch_type` | Previous pitch in at-bat |
| `last_pitch_category` | FB/BR/OS of last pitch |
| `last_pitch_in_zone` | Was last pitch a strike? |
| `last_pitch_swing` | Did batter swing? |
| `pitches_seen_this_ab` | Pitch count in at-bat |
| `fastball_pct_this_ab` | % fastballs seen so far |
| `same_pitch_streak` | Consecutive same pitch type |

### Situational Features

| Feature | Description |
|---------|-------------|
| `balls`, `strikes` | Count |
| `outs` | Outs in inning |
| `on_1b`, `on_2b`, `on_3b` | Base runners |
| `run_differential` | Score difference |
| `inning` | Current inning |
| `leverage_index` | Game situation importance |
| `platoon` | Same/opposite hand matchup |

### Pitch-Specific Features

| Feature | Description |
|---------|-------------|
| `pitch_type` | The pitch being evaluated |
| `pitch_category` | FB/BR/OS |
| `typical_velo` | Pitcher's avg velocity for this pitch |
| `typical_movement_x` | Horizontal movement |
| `typical_movement_z` | Vertical movement |
| `velo_diff_from_fb` | Speed differential from fastball |

## Model Architecture

### Two-Stage Approach

**Stage 1: Outcome Prediction**
Predict probability of each outcome:
- Ball
- Called strike
- Swinging strike
- Foul
- In play (with xwOBA for contact quality)

**Stage 2: Run Value Calculation**
Convert outcome probabilities to expected run value using:
- RE24 matrix for count changes
- xwOBA for contact outcomes
- Leverage-weighted for high-pressure situations

### Why Two Stages?

1. **Interpretability**: You can see WHY a pitch is recommended
   - "Slider recommended: 32% chase rate, only 0.280 xwOBA on contact"
   
2. **Debugging**: Easier to identify model failures
   - "Model thinks he'll chase 50% but he only chases 20%"
   
3. **Flexibility**: Different objectives for different situations
   - Strikeout pitch vs groundball pitch vs weak contact pitch

## Handling Sparse Data

### The Problem
Many batter-pitch matchups have few observations. A batter with 10 ABs vs sliders hitting .400 probably isn't actually a .400 slider hitter.

### The Solution: Hierarchical Shrinkage

```python
def get_batter_stat(batter_id, pitch_category, stat, min_sample=50):
    """Get batter stat with regression to mean for small samples."""
    
    batter_n = get_sample_size(batter_id, pitch_category)
    batter_stat = get_raw_stat(batter_id, pitch_category, stat)
    league_stat = get_league_average(pitch_category, stat)
    
    # Weight by sample size - more data = trust individual more
    weight = batter_n / (batter_n + min_sample)
    
    return weight * batter_stat + (1 - weight) * league_stat
```

This means:
- 100+ pitches seen: Mostly use individual stats
- 20-50 pitches: Blend individual and league average
- <20 pitches: Mostly use league average

## Evaluation Strategy

### Metrics

1. **Run Value Accuracy**: Does the model correctly predict which pitches yield lower run values?
   - Compare predicted RV ranking vs actual RV ranking
   
2. **Calibration**: Are the probabilities accurate?
   - If model says 30% chase rate, do they chase ~30% of the time?
   
3. **Decision Quality**: Do recommended pitches outperform random/naive?
   - Compare model picks vs "always throw best pitch" baseline
   - Compare vs actual pitcher decisions

### Time-Based Validation

**Critical**: Don't use random train/test split. Use time-based split:
- Train: April - July
- Validate: August
- Test: September

This simulates real deployment where you're predicting future at-bats.

## Files to Create

```
src/
├── fetch_statcast.py          # Download raw data
├── build_batter_profiles.py   # Aggregate batter vulnerability stats
├── build_pitcher_profiles.py  # Build pitcher arsenals
├── preprocess.py              # Clean data, add run values
├── feature_engineering.py     # Build ML features
├── train_model.py             # Train and evaluate model
├── predict.py                 # Make recommendations
└── utils.py                   # Shared utilities

data/
├── raw/
│   └── statcast_YYYY.csv
├── processed/
│   ├── batter_profiles.parquet
│   ├── pitcher_profiles.parquet
│   ├── pitches_with_rv.parquet
│   └── features.parquet
└── static/
    └── run_expectancy.csv

artifacts/
├── pitch_model.pkl
└── feature_columns.json
```

## Next Steps

1. Run `fetch_statcast.py` to get 2024 data (or use existing 2025 data)
2. Run pipeline scripts in order
3. Train model and evaluate
4. Integrate into Streamlit app
