"""
Train pitch recommendation model.

This script:
1. Loads engineered features
2. Splits data by time (not random) to avoid leakage
3. Trains a gradient boosting model to predict run value
4. Evaluates model performance
5. Saves model and evaluation metrics

Usage:
    python src/train_model.py
"""
import argparse
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent))

from utils import DATA_PROCESSED, ARTIFACTS

# ML imports
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib


def load_training_data():
    """Load features and prepare for training."""
    feature_path = DATA_PROCESSED / "features.parquet"
    
    if not feature_path.exists():
        raise FileNotFoundError(f"Features not found: {feature_path}. Run feature_engineering.py first.")
    
    print(f"Loading features from {feature_path}...")
    df = pd.read_parquet(feature_path)
    print(f"Loaded {len(df):,} samples")
    
    # Load feature column list
    feature_cols_path = ARTIFACTS / "feature_columns.json"
    if feature_cols_path.exists():
        with open(feature_cols_path) as f:
            feature_cols = json.load(f)
        feature_cols = [c for c in feature_cols if c in df.columns]
    else:
        exclude = ["game_pk", "game_date", "pitcher", "batter", 
                  "at_bat_number", "pitch_number", "run_value"]
        feature_cols = [c for c in df.columns if c not in exclude]
    
    print(f"Using {len(feature_cols)} features")
    return df, feature_cols


def time_based_split(df: pd.DataFrame, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split data by time to avoid leakage."""
    if "game_date" not in df.columns:
        print("Warning: No game_date column, using index-based split")
        n = len(df)
        train_end = int(n * train_frac)
        val_end = int(n * (train_frac + val_frac))
        return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]
    
    df = df.sort_values("game_date")
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    
    print(f"Train: {len(train_df):,} ({train_df['game_date'].min()} to {train_df['game_date'].max()})")
    print(f"Val:   {len(val_df):,} ({val_df['game_date'].min()} to {val_df['game_date'].max()})")
    print(f"Test:  {len(test_df):,} ({test_df['game_date'].min()} to {test_df['game_date'].max()})")
    
    return train_df, val_df, test_df


def identify_column_types(df: pd.DataFrame, feature_cols: list):
    """Identify numeric vs categorical columns."""
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col not in df.columns:
            continue
        if df[col].dtype in ['object', 'category', 'bool']:
            categorical_cols.append(col)
        elif df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
            numeric_cols.append(col)
        else:
            # Default to numeric
            numeric_cols.append(col)
    
    return numeric_cols, categorical_cols


def build_model_pipeline(numeric_cols: list, categorical_cols: list):
    """Build sklearn pipeline with preprocessing and model."""
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )
    
    # Full pipeline with model
    # Using HistGradientBoostingRegressor for speed and native missing value handling
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', HistGradientBoostingRegressor(
            max_depth=8,
            learning_rate=0.05,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        ))
    ])
    
    return model


def evaluate_model(model, X, y, set_name: str = ""):
    """Evaluate model and return metrics."""
    preds = model.predict(X)
    
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    print(f"\n{set_name} Metrics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "r2": r2, "mse": mse}


def evaluate_pitch_ranking(model, df: pd.DataFrame, feature_cols: list):
    """
    Evaluate if model correctly ranks pitches by run value.
    
    For each at-bat, check if the model's recommended pitch
    (lowest predicted run value) was actually better than average.
    """
    print("\nEvaluating pitch ranking quality...")
    
    # Group by at-bat
    ab_cols = ["game_pk", "at_bat_number"] if all(c in df.columns for c in ["game_pk", "at_bat_number"]) else None
    
    if ab_cols is None or "pitch_type" not in df.columns:
        print("  Cannot evaluate ranking (missing columns)")
        return {}
    
    # Get predictions
    X = df[feature_cols]
    df = df.copy()
    df["pred_rv"] = model.predict(X)
    
    # For each at-bat, find the pitch with lowest predicted RV
    results = []
    for _, group in df.groupby(ab_cols):
        if len(group) < 2:
            continue
        
        # Model's pick (lowest predicted RV)
        model_pick_idx = group["pred_rv"].idxmin()
        model_pick_actual_rv = group.loc[model_pick_idx, "run_value"]
        
        # Average actual RV for this at-bat
        avg_rv = group["run_value"].mean()
        
        # Did model pick better than average?
        model_better = model_pick_actual_rv < avg_rv
        
        results.append({
            "model_rv": model_pick_actual_rv,
            "avg_rv": avg_rv,
            "model_better": model_better
        })
    
    if not results:
        print("  No multi-pitch at-bats to evaluate")
        return {}
    
    results_df = pd.DataFrame(results)
    
    pct_better = results_df["model_better"].mean() * 100
    avg_improvement = (results_df["avg_rv"] - results_df["model_rv"]).mean()
    
    print(f"  Model pick better than avg: {pct_better:.1f}% of at-bats")
    print(f"  Average RV improvement: {avg_improvement:.4f}")
    
    return {
        "pct_better_than_avg": pct_better,
        "avg_rv_improvement": avg_improvement
    }


def get_feature_importance(model, numeric_cols: list, categorical_cols: list):
    """Extract feature importances from trained model."""
    
    # Get the regressor from pipeline
    regressor = model.named_steps['regressor']
    
    if not hasattr(regressor, 'feature_importances_'):
        return {}
    
    importances = regressor.feature_importances_
    
    # Get feature names after preprocessing
    preprocessor = model.named_steps['preprocessor']
    
    # Numeric features keep their names
    feature_names = list(numeric_cols)
    
    # Categorical features get expanded
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['encoder']
        if hasattr(cat_encoder, 'get_feature_names_out'):
            cat_names = cat_encoder.get_feature_names_out(categorical_cols)
            feature_names.extend(cat_names)
    
    # Match lengths
    if len(feature_names) != len(importances):
        print(f"Warning: Feature name mismatch ({len(feature_names)} names, {len(importances)} importances)")
        return {}
    
    # Create importance dict
    importance_dict = dict(zip(feature_names, importances))
    
    # Sort by importance
    sorted_importance = sorted(importance_dict.items(), key=lambda x: -x[1])
    
    return dict(sorted_importance[:30])  # Top 30


def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Training Pitch Recommendation Model")
    print("=" * 60)
    
    # Load data
    df, feature_cols = load_training_data()
    
    # Remove rows with missing target
    if "run_value" not in df.columns:
        raise ValueError("No run_value column in features")
    
    df = df[df["run_value"].notna()]
    print(f"After removing missing targets: {len(df):,} samples")
    
    # Split data
    train_df, val_df, test_df = time_based_split(df)
    
    # Identify column types
    numeric_cols, categorical_cols = identify_column_types(train_df, feature_cols)
    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Build model
    model = build_model_pipeline(numeric_cols, categorical_cols)
    
    # Prepare data
    X_train = train_df[feature_cols]
    y_train = train_df["run_value"]
    X_val = val_df[feature_cols]
    y_val = val_df["run_value"]
    X_test = test_df[feature_cols]
    y_test = test_df["run_value"]
    
    # Train
    print("\nTraining model...")
    model.fit(X_train, y_train)
    print("Training complete!")
    
    # Evaluate
    train_metrics = evaluate_model(model, X_train, y_train, "Train")
    val_metrics = evaluate_model(model, X_val, y_val, "Validation")
    test_metrics = evaluate_model(model, X_test, y_test, "Test")
    
    # Evaluate pitch ranking
    ranking_metrics = evaluate_pitch_ranking(model, test_df, feature_cols)
    
    # Feature importance
    print("\nTop Feature Importances:")
    importance = get_feature_importance(model, numeric_cols, categorical_cols)
    for i, (feat, imp) in enumerate(list(importance.items())[:15]):
        print(f"  {i+1}. {feat}: {imp:.4f}")
    
    # Save model
    model_path = ARTIFACTS / "pitch_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nSaved model to {model_path}")
    
    # Save metadata
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "ranking_metrics": ranking_metrics,
        "feature_importance": importance
    }
    
    metadata_path = ARTIFACTS / "model_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved metadata to {metadata_path}")
    
    # Save feature columns for inference
    inference_config = {
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }
    config_path = ARTIFACTS / "inference_config.json"
    with open(config_path, "w") as f:
        json.dump(inference_config, f, indent=2)
    print(f"Saved inference config to {config_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, metadata


if __name__ == "__main__":
    main()
