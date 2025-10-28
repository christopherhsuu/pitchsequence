import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt


PROCESSED_DIR = Path("data/processed")
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

NUM = [
    "balls","strikes","outs_when_up",
    "on_1b","on_2b","on_3b",
    "release_speed","release_spin_rate",
    "zone","plate_x","plate_z","last_pitch_speed_delta",
    "batter_pitchtype_woba","batter_pitchtype_whiff_rate","batter_pitchtype_run_value"
]
CAT = ["stand","p_throws","pitch_type","last_pitch_type","last_pitch_result"]

if __name__ == "__main__":
    # Load processed features and targets
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")

    # Load cluster-level averages (hitter archetypes) and merge when available
    cluster_path = Path("data/cluster_averages.csv")
    if cluster_path.exists():
        cluster_df = pd.read_csv(cluster_path)
        # prefix cluster features to avoid collisions
        cluster_df = cluster_df.rename(columns={c: f"cluster_{c}" for c in cluster_df.columns if c != 'cluster'})
        # ensure cluster column types match
        if 'cluster' in df.columns:
            df = df.merge(cluster_df, on='cluster', how='left')
        elif 'batter_cluster' in df.columns:
            df = df.merge(cluster_df, left_on='batter_cluster', right_on='cluster', how='left')
        else:
            # no cluster column present in processed features; try to join via archetypes if available
            # fallback: do nothing
            pass
    else:
        cluster_df = None

    y = df["run_value"]

    # Append any cluster-level numeric features to NUM
    cluster_features = []
    if cluster_df is not None:
        cluster_features = [c for c in df.columns if c.startswith('cluster_')]

    X = df[NUM + cluster_features + CAT]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([("num", num_pipe, NUM), ("cat", cat_pipe, CAT)])
    model = HistGradientBoostingRegressor(max_depth=8, learning_rate=0.06)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: {rmse:.4f}")
    # Save baseline model
    joblib.dump(pipe, ARTIFACTS_DIR / "runvalue_model.pkl")

    # Also save clustered model if cluster features were used
    if cluster_features:
        clustered_model_path = ARTIFACTS_DIR / "pitchsequence_clustered_model.pkl"
        joblib.dump(pipe, clustered_model_path)

        # Attempt to compute and save feature importances for interpretability
        try:
            import numpy as _np
            import matplotlib.pyplot as plt

            # extract feature names from ColumnTransformer
            pre = pipe.named_steps['pre']
            # get feature names for numeric and categorical
            num_features = NUM + cluster_features
            # for categorical, get names from OneHotEncoder if present
            cat_features = []
            if hasattr(pre, 'transformers_'):
                # transformers_ is a list of tuples
                for name, trans, cols in pre.transformers_:
                    if name == 'cat' and hasattr(trans, 'named_steps'):
                        ohe = trans.named_steps.get('ohe')
                        if ohe is not None:
                            try:
                                ohe_names = ohe.get_feature_names_out(cols)
                                cat_features = list(ohe_names)
                            except Exception:
                                # fallback
                                cat_features = list(cols)
            feature_names = list(num_features) + list(cat_features)

            model = pipe.named_steps['model']
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                # align lengths
                if len(importances) == len(feature_names):
                    idx = _np.argsort(importances)[::-1][:30]
                    top_feats = [feature_names[i] for i in idx]
                    top_imp = importances[idx]
                    plt.figure(figsize=(8, 6))
                    plt.barh(range(len(top_imp))[::-1], top_imp, color='tab:blue')
                    plt.yticks(range(len(top_imp))[::-1], top_feats)
                    plt.xlabel('Importance')
                    plt.title('Top feature importances (cluster-aware model)')
                    plt.tight_layout()
                    plt.savefig(ARTIFACTS_DIR / 'cluster_feature_importance.png')
                    plt.close()
        except Exception:
            # don't fail training for plotting errors
            pass
