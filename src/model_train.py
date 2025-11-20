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
    df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
    y = df["run_value"]
    X = df[NUM + CAT]

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
    joblib.dump(pipe, ARTIFACTS_DIR / "runvalue_model.pkl")
