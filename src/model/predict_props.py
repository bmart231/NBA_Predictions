# Take a player + matchup (home/away) and predict over/under prop for that player
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd 
import scipy.stats as stats
import joblib

features_path = Path("data/processed/player_features.parquet")
model_path = Path("models")

# standard normal cumulative distribution function
def normal_cdf(z: float) -> float:
    # compute the cdf using erf
    return float(stats.norm.cdf(z))

def predict_player_prop(mu: float, sigma: float, line: float) -> tuple[float, float]:
    """Returns probability of going over and under the line given a normal distribution."""
    sigma = max(float(sigma), 1e-6)  # prevent division by zero
    z = (line - mu) / sigma # z-score
    p_under = normal_cdf(z) # probability of going under the line
    p_over = 1.0 - p_under # probability of going over the line
    return p_over, p_under # return as tuple

# load a trained model and its feature columns
def load_model(which: str):
    path = model_path / f"{which}_mean.joblib" # load model file path
    bundle = joblib.load(path) # load the model bundle 
    return bundle["model"], bundle["features"] # return model and feature columns 

def latest_features_for_players(df: pd.DataFrame, player_ids: str) -> pd.Series:
    """Get the latest features for a list of player IDS."""
    subset = df[df["PLAYER_ID"].str.lower() == player_ids.lower()].copy() # filter by player ID
    if subset.empty:
        raise ValueError(f"No features found for Player: {player_ids}")
    
    subset = subset.sort_values("Game_Date", ascending=False) # sort by most recent game date

    return subset.iloc[0] # return the most recent row of features

def main() -> None:
    # Player ID
    
    player_id = "Lebron James"
    pts_line = 27.5
    reb_line = 7.5
    
    # choose which rolling std to use as a sigma estimate
    pts_sigma_col = "PTS_roll_std_10"
    reb_sigma_col = "REB_roll_std_10"
    
    df = pd.read_parquet(features_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    row = latest_features_for_players(df, player_id)

    pts_model, feature_cols = load_model("pts")
    reb_model, feature_cols2 = load_model("reb")

    # (sanity) both models should use same features
    if feature_cols != feature_cols2:
        raise RuntimeError("Feature lists differ between pts and reb models.")

    X = row[feature_cols].to_frame().T  # shape (1, p)

    mu_pts = float(pts_model.predict(X)[0])
    mu_reb = float(reb_model.predict(X)[0])

    sigma_pts = float(row.get(pts_sigma_col, 0.0))
    sigma_reb = float(row.get(reb_sigma_col, 0.0))

    p_over_pts, p_under_pts = predict_player_prop(mu_pts, sigma_pts, pts_line)
    p_over_reb, p_under_reb = predict_player_prop(mu_reb, sigma_reb, reb_line)

    print(f"Player: {player_id}")
    print(f"Using latest game date: {row['GAME_DATE'].date()}")

    print("\nPOINTS")
    print(f"  predicted mean (mu): {mu_pts:.2f}")
    print(f"  predicted sigma:     {sigma_pts:.2f}  (from {pts_sigma_col})")
    print(f"  line:                {pts_line}")
    print(f"  P(Over):             {p_over_pts:.3f}")
    print(f"  P(Under):            {p_under_pts:.3f}")

    print("\nREBOUNDS")
    print(f"  predicted mean (mu): {mu_reb:.2f}")
    print(f"  predicted sigma:     {sigma_reb:.2f}  (from {reb_sigma_col})")
    print(f"  line:                {reb_line}")
    print(f"  P(Over):             {p_over_reb:.3f}")
    print(f"  P(Under):            {p_under_reb:.3f}")


if __name__ == "__main__":
    main()
    

