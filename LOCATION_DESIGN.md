# Pitch Location Recommendation Design

## Problem Statement

Currently, pitch locations are hardcoded and ignore context:
- Changeups always recommended "middle" (terrible!)
- No consideration of batter tendencies
- No adaptation to count/situation

**Goal**: Recommend both pitch TYPE and optimal LOCATION based on:
1. Pitch characteristics (movement, velocity)
2. Batter vulnerabilities by zone
3. Count and situation
4. Sequence context

## Data Analysis

### Key Findings from 2024 Statcast Data

**Location coordinates**:
- `plate_x`: -3 to +3 feet (negative = inside to RHH, positive = outside)
- `plate_z`: 1 to 4 feet (height from ground)
- `zone`: 1-9 = strike zone grid, 11-14 = out of zone

**Run Value by Location** (all pitches):
- **Inside**: +0.003 RV
- **Middle**: +0.001 RV
- **Outside**: -0.003 RV ← **Best**
- **Low**: -0.008 RV ← **Best**
- **Middle-height**: +0.004 RV
- **High**: +0.003 RV

**Pitch-Specific Patterns**:

| Pitch | Best Horizontal | Best Vertical | Worst Location |
|-------|-----------------|---------------|----------------|
| FF (Fastball) | Outside edges | High | Middle-low |
| SL (Slider) | Outside-outside | Low | Middle |
| CH (Changeup) | Edges | Low | Middle-high |
| CU (Curveball) | Low edge | Very low | Middle |
| SI (Sinker) | Inside/Outside | Low | High |

**Key Insight**: Middle of plate is almost always BAD, especially for offspeed pitches.

## Proposed Solution: Two-Stage Model

### Stage 1: Pitch Type Selection (Already Built)
Current model selects optimal pitch type (FF, SL, CH, etc.)

### Stage 2: Location Optimization (New)

**Architecture**:
```
For each recommended pitch type:
  Grid search over candidate locations (e.g., 5x5 grid in strike zone)
  For each location:
    features = [pitch_type, plate_x, plate_z, batter_zone_stats, situation]
    predicted_rv = location_model.predict(features)

  Return location with minimum predicted RV
```

**Model Input Features**:
1. **Pitch characteristics**:
   - `pitch_type`, `pitch_category`
   - `pitcher_avg_velo`, `pitcher_avg_pfx_x`, `pitcher_avg_pfx_z`
   - `pitcher_whiff_rate` (for this pitch)

2. **Location being evaluated**:
   - `plate_x`, `plate_z`
   - `zone` (1-14 grid)
   - `is_in_zone` (binary)

3. **Batter zone vulnerabilities** (NEW - need to build):
   - `batter_chase_rate_edge` (swings at pitches off plate)
   - `batter_zone_1_woba`, `batter_zone_2_woba`, ... (9 zones)
   - `batter_low_ball_contact_rate`
   - `batter_high_ball_whiff_rate`

4. **Situational**:
   - Count, outs, runners
   - Platoon matchup
   - Sequence context

5. **Interaction features**:
   - `batter_weakness_at_location` (does batter struggle here?)
   - `pitch_effectiveness_at_location` (does THIS pitch work here?)

**Output**: Predicted run value for throwing this pitch at this location

### Implementation Steps

#### 1. Build Batter Zone Profiles
```python
# src/build_batter_zone_profiles.py

# Aggregate batter stats by zone
for zone in 1-14:
    whiff_rate_zone_{zone}
    contact_rate_zone_{zone}
    xwoba_zone_{zone}
    chase_rate (zones 11-14)
```

#### 2. Add Location Features to Training Data
```python
# src/feature_engineering.py - add location features

def add_location_features(df):
    # Discretize locations for easier modeling
    df['loc_x_category'] = pd.cut(df['plate_x'],
        bins=[-5, -1.0, -0.3, 0.3, 1.0, 5],
        labels=['inside-inside', 'inside', 'middle', 'outside', 'outside-outside'])

    df['loc_z_category'] = pd.cut(df['plate_z'],
        bins=[0, 1.8, 2.5, 3.5, 6],
        labels=['low', 'mid-low', 'mid-high', 'high'])

    # Batter vulnerability at this location
    # Join zone-specific stats
    df = join_batter_zone_stats(df)

    # Pitcher effectiveness at this location
    # (already have pitch characteristics)

    return df
```

#### 3. Train Location-Aware Model
```python
# src/train_location_model.py

# Option A: Single model with location features
features = [
    pitch_features + location_features +
    batter_zone_features + situation_features
]
target = run_value

model = GradientBoostingRegressor()
model.fit(features, target)

# Option B: Separate model per pitch type
for pitch_type in ['FF', 'SL', 'CH', 'CU', ...]:
    data = training_data[training_data['pitch_type'] == pitch_type]
    model[pitch_type] = train(data)
```

#### 4. Location Optimization at Inference
```python
# src/predict.py - add location recommendation

def recommend_with_location(pitcher_id, batter_id, situation):
    # Step 1: Get top pitch types (existing)
    top_pitches = recommend_pitch_type(...)  # e.g., ['SL', 'FF', 'CH']

    # Step 2: For each pitch, find optimal location
    recommendations = []
    for pitch in top_pitches:
        best_location = optimize_location(
            pitch_type=pitch,
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            situation=situation
        )
        recommendations.append({
            'pitch': pitch,
            'location_x': best_location['plate_x'],
            'location_z': best_location['plate_z'],
            'zone': best_location['zone'],
            'expected_rv': best_location['rv']
        })

    return recommendations

def optimize_location(pitch_type, batter_id, pitcher_id, situation):
    # Grid search over strike zone
    candidate_locations = [
        # 5x5 grid covering strike zone + edges
        {'plate_x': x, 'plate_z': z, 'zone': get_zone(x, z)}
        for x in [-1.2, -0.6, 0.0, 0.6, 1.2]  # horizontal
        for z in [1.8, 2.3, 2.8, 3.3, 3.8]    # vertical
    ]

    best_location = None
    best_rv = float('inf')

    for loc in candidate_locations:
        # Build feature vector
        features = build_location_features(
            pitch_type=pitch_type,
            plate_x=loc['plate_x'],
            plate_z=loc['plate_z'],
            batter_id=batter_id,
            pitcher_id=pitcher_id,
            situation=situation
        )

        # Predict run value at this location
        predicted_rv = location_model.predict([features])[0]

        if predicted_rv < best_rv:
            best_rv = predicted_rv
            best_location = loc

    best_location['rv'] = best_rv
    return best_location
```

## Alternative: Rule-Based System (Quick Fix)

For immediate improvement without retraining:

```python
# Pitch type -> optimal location rules (from data analysis)
PITCH_LOCATION_RULES = {
    'FF': {
        'horizontal': 'outside',     # or 'up-and-in' for inside approach
        'vertical': 'high',
        'avoid': 'middle-low'
    },
    'SI': {
        'horizontal': 'inside-or-outside',
        'vertical': 'low',
        'avoid': 'high-middle'
    },
    'SL': {
        'horizontal': 'outside',     # chase pitch
        'vertical': 'low',
        'avoid': 'middle'
    },
    'CU': {
        'horizontal': 'middle-outside',
        'vertical': 'low',           # drop it in
        'avoid': 'middle-middle'
    },
    'CH': {
        'horizontal': 'outside',     # away from RHH
        'vertical': 'low',
        'avoid': 'middle-high'       # CRITICAL: changeup middle-high is batting practice
    },
    'FS': {
        'horizontal': 'outside',
        'vertical': 'low',
        'avoid': 'middle'
    },
}

def get_rule_based_location(pitch_type, batter_hand):
    rule = PITCH_LOCATION_RULES.get(pitch_type, {'horizontal': 'outside', 'vertical': 'low'})

    # Adjust for batter handedness
    if batter_hand == 'L':
        if rule['horizontal'] == 'outside':
            return 'away-low'  # away from LHH = positive plate_x
        elif rule['horizontal'] == 'inside':
            return 'in-low'    # inside to LHH = negative plate_x
    else:  # RHH
        if rule['horizontal'] == 'outside':
            return 'away-low'  # away from RHH = positive plate_x
        elif rule['horizontal'] == 'inside':
            return 'in-high'   # inside to RHH = negative plate_x

    return f"{rule['horizontal']}-{rule['vertical']}"
```

## Evaluation Metrics

1. **Location Quality**: Does recommended location have lower RV than random?
2. **Strike Rate**: Are recommended locations in/near strike zone?
3. **Batter Exploitation**: Do recommendations target batter's weak zones?
4. **Realism**: Do recommendations match what actual MLB pitchers do?

Compare:
- Random location
- Rule-based location
- ML-optimized location
- Actual MLB pitcher decisions

## Next Steps

1. **Immediate**: Implement rule-based system (1-2 hours)
2. **Short-term**: Build batter zone profiles (2-3 hours)
3. **Medium-term**: Add location features to training data (2-3 hours)
4. **Long-term**: Train location-aware model and grid search optimizer (4-6 hours)

## Example Output

Before (current):
```
Recommendation: Changeup, middle, 35% probability
→ Terrible! Changeup middle-middle is RV = +0.007 (bad for pitcher)
```

After (with location model):
```
Recommendation: Changeup, low-and-away, 35% probability
Location: plate_x = +0.8, plate_z = 2.1 (zone 9)
Expected RV: -0.014
Reasoning: Batter chases low-away breaking balls 38% of time,
           only makes contact 65% when they do
```
