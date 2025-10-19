# PitchSequence — Pitch-sequence recommender and run-value model

This repository contains a reproducible pipeline and interactive Streamlit app that recommends next pitches to throw to a batter based on historical data, derived batter archetypes, pitcher arsenals, and a run-value model trained on play-level outcomes. The project combines data preparation, feature engineering, machine learning, and a lightweight UI to explore pitch sequencing from a sabermetric perspective.

## Project overview

- Goal: recommend the next pitch (and ranked candidate list) for a pitcher vs. batter matchup in a given count and game situation, using a trained run-value model and batter-pitchtype matchup stats.
- Data sources: Statcast play-by-play (raw/processed), derived batter archetypes, inferred pitcher arsenals, and a run-expectancy lookup table.
- Unique aspects:
  - Stepwise, name-driven Streamlit UI that accepts names only for batter and pitcher and manages an at-bat flow (choose pitch, record outcome, repeat).
  - Uses run-expectancy to convert model predicted run-values into softmax probabilities to prioritize lower expected runs.
  - Combines a model-based scoring approach with heuristic fallbacks and batter-specific pitchtype stats for robustness.

## Quick start — run locally

1. Create a Python 3.13 virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit app locally:

```bash
streamlit run app.py
```

3. Open http://localhost:8501 in your browser.

If you deploy to Streamlit Cloud, ensure the "main module" path in the deployment settings is set to `app/app.py` (this repo provides a tiny wrapper at that location that imports the canonical `app.py`).

## Data preparation

Files & directories:
- `data/raw/` — raw data dumps (example: `statcast_2025.csv`, and name mapping CSVs for batters/pitchers)
- `data/processed/` — parquet files with cleaned features and aggregated batter-pitchtype stats (`cleaned.parquet`, `features.parquet`, `runvalue.parquet`)
- `pitcher_assets/pitcher_arsenals.csv` — inferred pitcher arsenals built from Statcast

Steps performed:

1. Acquire raw Statcast plays and name mapping files. Place raw CSVs in `data/raw/`.
2. Run `src/preprocess.py` to clean raw play-by-play rows and filter to the seasons and events of interest.
   - Tasks performed: parse timestamps, normalize pitch types, drop clearly invalid rows (missing pitch_type + outcome), and standardize batter/pitcher name fields.
3. Run `src/feature_engineering.py` to compute per-play features and aggregate batter vs. pitch_type summary stats.
   - Feature tasks: extract plate_x/plate_z approximations, map human-friendly location tokens to plate coordinates for the model when a visual target is used, compute last-pitch deltas, and bucket zones.
4. Handle missing values and outliers:
   - Numeric missing values are filled with domain-appropriate defaults (e.g., 0 for rate stats when absent, or median imputation for continuous measures) inside the feature pipeline.
   - Extreme outliers are winsorized or clipped where appropriate (for example, release speeds outside realistic ranges are clipped to min/max bounds).
5. Save processed data to Parquet under `data/processed/` for efficient re-use during model training.

Filtering decisions and heuristics:
- Only include pitchers/batters with sufficient counts for stable estimates (e.g., minimum plate appearances or pitch-type exposures). See the processing scripts for exact thresholds used.
- Limit seasons or games as needed for a stable modeling window (e.g., last 1–3 seasons depending on dataset size).

## Feature selection & engineering

Primary features used by the run-value model and UI:
- Count and base-state features: balls, strikes, outs_when_up, on_1b, on_2b, on_3b
- Pitch characteristics: pitch_type (categorical), release_speed, release_spin_rate
- Plate and zone features: plate_x, plate_z, zone
- Temporal features: last_pitch_type, last_pitch_result, last_pitch_speed_delta
- Batter matchup aggregates: batter_pitchtype_woba, batter_pitchtype_whiff_rate, batter_pitchtype_run_value — derived from aggregating `features.parquet` by batter and pitch type

Transformations and preprocessing:
- Categorical variables are encoded via one-hot/target-encoding inside the model pipeline.
- Numeric variables are scaled when required by the model (the training pipeline contains scalers where necessary).
- Feature interactions (e.g., count x pitch_type) are created to help the model capture contextual pitch selection effects.

Rationale for chosen features:
- The chosen features capture both the physical context (release speed, zone) and matchup context (batter-specific pitchtype rates) that drive run expectancy.
- Using batter-specific aggregated pitchtype run-values allows the model to account for individual tendencies beyond league averages.

## Model training process

Files of interest:
- `src/model_train.py` — training script that builds the modeling pipeline and saves the artifact
- `artifacts/runvalue_model.pkl` — trained scikit-learn pipeline and model (used for scoring in the Streamlit app)

Workflow summary:

1. Load processed features (`data/processed/features.parquet`) and the `runvalue` labels (`data/processed/runvalue.parquet`).
2. Split data into train/validation/test sets by game or time window to avoid leakage (time-based split recommended).
3. Build a scikit-learn pipeline that handles categorical encoding, imputation, scaling, and the estimator.
   - Example model choices: GradientBoostingRegressor or RandomForestRegressor for run-value prediction. In some experiments a multi-output approach was used for predicting multiple outcome components.
4. Tune hyperparameters using grid search or randomized search with cross-validation on the train set.
5. Evaluate on the validation set and select the best model.
6. Re-fit on train+validation and test once on the holdout test set. Save the final pipeline using `joblib.dump` or `pickle` to `artifacts/runvalue_model.pkl`.

Notes:
- When saving models, be mindful of library versions (scikit-learn). If your deployment environment uses a different scikit-learn version you may see compatibility warnings when unpickling.

## Evaluation

Metrics and methods used:
- For run-value regression: mean squared error (MSE), mean absolute error (MAE), and explained variance.
- For derived decision-support quality: evaluate whether the model's ranking of pitch candidates correlates with lower realized run outcomes in holdout data.
- Cross-validation: time-aware cross-validation (sliding window) is used to avoid leakage across games/seasons.
- Diagnostics: residual plots, feature importances, and partial dependence plots were used to understand model behavior.

## Model deployment & usage

How the model is used in practice:
- The Streamlit app (`app.py`) loads `artifacts/runvalue_model.pkl` at runtime (if present) and uses it to score every pitch in the pitcher's arsenal for the current state. The app also builds a small `batter_stats_map` from batter vs. pitch_type aggregates so predictions are batter-aware.
- The raw model predictions are converted into candidate probabilities by computing expected_after_RE = current_RE + predicted_run_value, then applying a softmax over the negative expected_after_RE to prefer lower expected runs.

UI flow:
1. Choose batter (names-only) and pitcher (names-only), set the count and base occupancy, click "Start at-bat".
2. The app computes candidate pitches (model-based when possible, heuristic otherwise) and renders them with percent probabilities and small strike-zone visuals.
3. Select a pitch, record the outcome, and the session updates the count and recomputes candidates. This lets you step through an at-bat interactively.

Deployment notes:
- On Streamlit Cloud: set the repository and branch, and set the main module to `app/app.py` (the wrapper imports the canonical `app.py`). If you prefer, configure the host to use `app.py` directly and delete the wrapper.
- If you see scikit-learn InconsistentVersionWarning when loading the model, consider re-saving the model with the environment's scikit-learn or pinning scikit-learn in `requirements.txt` to the training version.

## Project file map (what to look at)

- `app.py` — main Streamlit UI and session management (stepwise at-bat flow, candidate rendering, model integration)
- `app/app.py` — minimal wrapper imported by some hosts (imports root `app.py`) for compatibility with hosting entrypoint settings
- `src/attack_recommender.py` — heuristic candidate generator and batter-pitchtype stat helpers
- `src/predict.py` — model wrapper and scoring utilities (loads `artifacts/runvalue_model.pkl`)
- `src/model_train.py` — training script for building the pipeline and saving models
- `src/feature_engineering.py` — functions that compute play-level features and batter-pitchtype aggregates
- `src/preprocess.py` — cleaning and normalization of raw Statcast play-by-play
- `src/build_pitcher_arsenals.py` — infer pitcher arsenals from Statcast and write `pitcher_assets/pitcher_arsenals.csv`
- `data/raw/` — raw CSVs (statcast, mapping files)
- `data/processed/` — parquet outputs used for training (`cleaned.parquet`, `features.parquet`, `runvalue.parquet`)
- `artifacts/runvalue_model.pkl` — trained model artifact used in production
- `pitcher_assets/pitcher_arsenals.csv` — inferred arsenals for pitchers

## Reproducing the training pipeline

1. Ensure raw data is available in `data/raw/` and the mapping CSVs for names are present.
2. Run preprocessing and feature engineering (order matters):

```bash
python src/preprocess.py --input data/raw/statcast_2025.csv --output data/processed/cleaned.parquet
python src/feature_engineering.py --input data/processed/cleaned.parquet --output data/processed/features.parquet
```

3. Train the model:

```bash
python src/model_train.py --features data/processed/features.parquet --labels data/processed/runvalue.parquet --out artifacts/runvalue_model.pkl
```

4. Start the app and confirm it loads the saved model and artifacts.

If any step fails because of package version issues (e.g., scikit-learn), create a virtualenv with pinned versions or re-run training in the deployment environment to produce a compatible artifact.

## Future improvements

- Add more batter-level features, including Statcast swing/contact metrics (swinging_strike_rate, chase_rate, etc.)
- Model ensemble or stacking to improve run-value predictions
- Re-train model periodically or with online updates to capture season drift
- Create a richer visualization dashboard showing sequence-level outcomes and aggregated evaluation (e.g., which pitch sequences lower run expectancy vs. holdout)
- Add user authentication and session persistence to save at-bats for later analysis

## Contact & acknowledgements

If you have questions or want to reproduce the pipeline, open an issue or contact the maintainer (repository owner). Thanks to the Statcast data providers and the open-source Python ecosystem for enabling reproducible analytics.
## Deploying to Vercel (serverless API)

This repository includes a FastAPI serverless endpoint at `api/recommend.py` that wraps the recommender logic.

Quick steps to deploy:

1. Install the Vercel CLI and log in.
2. Push this repository to GitHub (or connect your repo in the Vercel dashboard).
3. Vercel will detect the Python serverless functions under `api/` and install packages from `requirements.txt`.

Endpoint:

POST /.vercel.app/api/recommend (or `https://<your-deployment>.vercel.app/api/recommend`)

Example JSON body:

{
	"batter": "455117",
	"pitcher": "123456",
	"count": "1-1",
	"situation": {"risp": true, "outs": 2, "late_inning": true}
}

The endpoint returns a JSON recommendation (recommended_sequence, confidence, strategy_notes, ...).

Notes:
- Vercel serverless functions are short-lived. This API is stateless and loads data on startup.
- The Streamlit app in `app/app.py` is not suitable for Vercel serverless; consider using the `/api/recommend` endpoint from a lightweight static frontend or another deployment for Streamlit.
# pitchsequence!
