# PitchSequence Redesign - Context for Claude

## What This Is

This is a redesigned pitch recommendation system. The goal is to recommend what pitch a pitcher should throw to minimize expected run value, accounting for:

1. **Batter-specific vulnerabilities** (not just archetypes - actual stats on what they can't hit)
2. **Pitch sequencing effects** (what was thrown before matters)
3. **Game situation context** (count, runners, outs, etc.)

## Key Design Decisions

### Why Not Clustering?

The old approach clustered batters by batting style (power hitter, contact hitter). But this doesn't map to pitch vulnerability. A "power hitter" tells you HOW they hit, not WHAT they can't hit.

### New Approach: Direct Vulnerability Features

Instead of clusters, we aggregate each batter's actual performance vs pitch categories:

- **FB** (Fastballs): FF, SI, FC, FT
- **BR** (Breaking): SL, CU, KC, ST, SV  
- **OS** (Offspeed): CH, FS, SC

For each batter + category, we calculate:
- `whiff_rate` - How often they swing and miss
- `chase_rate` - How often they chase out of zone
- `zone_contact_rate` - Can they make contact in zone?
- `xwoba` - Quality of contact when they connect
- `gb_rate` - Ground ball tendency

### Regression to Mean

Small samples are unreliable. A batter with 10 ABs hitting .400 vs sliders probably isn't a .400 slider hitter. We blend individual stats with league averages weighted by sample size:

```python
weight = sample_size / (sample_size + reliability_threshold)
estimate = weight * observed + (1 - weight) * league_average
```

## File Structure

```
src/
├── utils.py                 # Shared utilities, constants, helper functions
├── fetch_statcast.py        # Download raw data from Baseball Savant
├── build_batter_profiles.py # Aggregate batter vulnerability stats
├── build_pitcher_profiles.py# Build pitcher arsenals with characteristics
├── preprocess.py            # Clean data, calculate run values
├── feature_engineering.py   # Join profiles, create ML features
├── train_model.py           # Train and evaluate model
└── predict.py               # PitchRecommender class for inference

data/
├── raw/                     # Raw Statcast CSVs
├── processed/               # Parquet files with processed data
└── static/                  # Static reference data (run expectancy)

artifacts/
├── pitch_model.pkl          # Trained sklearn pipeline
├── model_metadata.json      # Training metrics, feature importance
└── inference_config.json    # Feature columns for prediction
```

## How to Run

1. **Fetch data** (if needed):
   ```bash
   python src/fetch_statcast.py --year 2024
   ```

2. **Run full pipeline**:
   ```bash
   python run_pipeline.py --skip-fetch  # if you already have data
   ```

3. **Or run steps individually**:
   ```bash
   python src/build_batter_profiles.py --input data/raw/statcast_2024.csv
   python src/build_pitcher_profiles.py --input data/raw/statcast_2024.csv
   python src/preprocess.py --input data/raw/statcast_2024.csv
   python src/feature_engineering.py
   python src/train_model.py
   ```

## Using the Recommender

```python
from src.predict import PitchRecommender

recommender = PitchRecommender()
results = recommender.recommend(
    pitcher_id=543037,
    batter_id=545361,
    balls=1,
    strikes=2,
    outs=1,
    on_1b=True,
    on_2b=False,
    on_3b=False,
    stand="R"
)

for r in results:
    print(f"{r['pitch_type']}: {r['probability_pct']:.1f}% - {r['reasoning']}")
```

## What Still Needs Work

1. **Integration with Streamlit app** - The new `predict.py` needs to be integrated into `app.py`

2. **More sequence features** - Currently limited; could add:
   - Full pitch history in at-bat
   - Pitcher fatigue indicators
   - Recent pitch mix

3. **Location recommendations** - Currently only recommends pitch TYPE, not location

4. **Two-stage outcome model** - Could predict P(strike), P(whiff), P(contact) separately for better interpretability

5. **Evaluation** - Need to evaluate on holdout data whether recommendations actually yield lower run values

## Questions to Consider

- Should we predict outcome probabilities first, then convert to run value?
- How do we handle pitchers the model hasn't seen?
- Should location be a separate model or integrated?
- How much historical data is enough? (1 season? 3 seasons?)
