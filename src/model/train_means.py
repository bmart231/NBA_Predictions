from __future__ import annotations
from pathlib import Path

import joblib # used to save and load models
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error

data_path = Path("data/processed/player_features_2023_2026.parquet")
model_path = Path("models")

n_splits = 5  # number of folds for cross-validation

# this functions returns the list of feature columns names to be used for training
# includes basic context features and rolling statistics 
def get_features_col(df: pd.DataFrame) -> list[str]:
    """Get the list of feature columns for training."""
    roll_mean_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]
    base_cols = ["IS_HOME", "IS_AWAY", "rest_days"]
    return base_cols + roll_mean_cols 

def train_one(df: pd.DataFrame, target_column: str, feature_cols: list[str]):
    df = df.sort_values("GAME_DATE").copy()
    X = df[feature_cols]
    y = df[target_column]

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes: list[float] = []

    for train_idx, val_idx in tscv.split(X):
        model = LinearRegression()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[val_idx])
        maes.append(mean_absolute_error(y.iloc[val_idx], pred))

    # final fit on all data
    final_model = LinearRegression()
    final_model.fit(X, y)

    cv_mae = float(sum(maes) / len(maes))
    return final_model, cv_mae

def main() -> None:
    model_path.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(data_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    
    features_col = get_features_col(df)
    # Train model for points
    pts_model, pts_mae = train_one(df, "y_pts", features_col)
    reb_model, reb_mae = train_one(df, "y_reb", features_col)
    
    joblib.dump({"model": pts_model, "features": features_col}, model_path / "pts_mean.joblib")
    joblib.dump({"model": reb_model, "features": features_col}, model_path / "reb_mean.joblib")
    
    
    print(f"PTS MAE (CV avg): {pts_mae:.3f}") # print mean absolute error for points model
    print(f"REB MAE (CV avg): {reb_mae:.3f}") # print mean absolute error for rebounds model
    print(f"Models saved to {model_path}")
    
    # to run
    
if __name__ == "__main__":
    main()
    


    
    

    
    

    

    


