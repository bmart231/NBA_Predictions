# Take a player + matchup (home/away) and predict over/under prop for that player
from __future__ import annotations
from html import parser
from pathlib import Path
import sys
import numpy as np
import pandas as pd 
import scipy.stats as stats
import joblib

import argparse

features_path = Path("data/processed/player_features_2023_2026.parquet")
model_path = Path("models")

# standard normal cumulative distribution function
def normal_cdf(z: float) -> float:
    # compute the cdf using erf
    return float(stats.norm.cdf(z))

def predict_player_prop(mu: float, sigma: float, line: float) -> tuple[float, float]:
    """Returns probability of going over and under the line given a normal distribution."""
    sigma = max(sigma, 1e-6)  # prevent division by zero
    z = (line - mu) / sigma # z-score
    p_under = normal_cdf(z) # probability of going under the line
    p_over = 1.0 - p_under # probability of going over the line
    return p_over, p_under # return as tuple

# load a trained model and its feature columns
def load_model(which: str):
    path = model_path / f"{which}_mean.joblib" # load model file path
    bundle = joblib.load(path) # load the model bundle 
    return bundle["model"], bundle["features"] # return model and feature columns 

def latest_features_for_players(df: pd.DataFrame, player_name: str) -> pd.Series:
    """Get the latest features for a list of player IDS."""
    subset = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy() # filter by player ID
    if subset.empty:
        raise ValueError(f"No features found for Player: {player_name}")
    
    subset = subset.sort_values("GAME_DATE", ascending=False) # sort by most recent game date
    return subset.iloc[0] # return the most recent row of features


# new features to include prob to american odds
def prob_to_american_odds(p: float) -> int:
    """"Convert probability p to American odds Format."""
    # ensure p is a float
    p = float(p)
    if p <= 0.0: # prevent invalid probabilities
        return 10_000  # "infinite" odds
    if p >= 1.0:
        return -10_000
    if p >= 0.5:
        # negative odds
        odds = - (p / (1 - p)) * 100
    else:
        # positive odds
        odds = ((1 - p) / p) * 100
        
    return int(round(odds))

def american_to_profit_per_1(odds: float) -> float:
    """
    Profit (not including stake) per $1 risked at American odds.
    -110 -> profit = 1/1.10 = 0.909...
    +150 -> profit = 1.50
    """
    o = float(odds)
    if o == 0:
        raise ValueError("American odds cannot be 0")
    if o > 0:
        return o / 100.0
    else:
        return 100.0 / abs(o)


def expected_value_per_1(p_win: float, odds: float) -> float:
    """Expected value per $1 bet given probability of winning and American odds.
    """
    profit = american_to_profit_per_1(odds)
    return float(p_win) * profit - (1.0 - float(p_win))

def argparse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NBA player prop over/under probabilities.")
    parser.add_argument("--player", help='Player name, e.g. "LeBron James"')
    parser.add_argument("--pts", type=float, help="Points line, e.g. 27.5")
    parser.add_argument("--reb", type=float, help="Rebounds line, e.g. 7.5")
    parser.add_argument("--sigma-window", type=int, default=20, choices=[5, 10, 20],
                        help="Rolling window for sigma estimate (5, 10, or 20). Default: 20")

    # to add sportsbook odds inputs
    parser.add_argument("--over-odds", type=float, default=None, help="Sportsbook odds for OVER (e.g., -110)")
    parser.add_argument("--under-odds", type=float, default=None, help="Sportsbook odds for UNDER (e.g., -110)")

    return parser.parse_args()


def main() -> None:
    args = argparse_args()

    # Prompt if missing
    args.player = input("Player name (e.g., LeBron James): ").strip()
    if not args.player:
        raise ValueError("Player name is required.")
    if not any(c.isalpha() for c in args.player):
        raise ValueError("Player name must contain letters. Please enter a valid name.")

    if args.pts is None:
        args.pts = float(input("Points line (e.g., 27.5): ").strip())

    if args.reb is None:
        args.reb = float(input("Rebounds line (e.g., 7.5): ").strip())
        
    if args.over_odds is None:
        args.over_odds = float(input("Over odds (American, e.g., -110 or +150): ").strip())

    if args.under_odds is None:
        args.under_odds = float(input("Under odds (American, e.g., -110 or +150): ").strip())
        
        
    sigma_window = args.sigma_window
    pts_sigma_col = f"PTS_roll{sigma_window}_std"
    reb_sigma_col = f"REB_roll{sigma_window}_std"

    df = pd.read_parquet(features_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str)
    
    # to filter to current season only
    # df = df[df["SEASON"] == "2025-26"].copy() doesn't work yet bruh 


    row = latest_features_for_players(df, player_name=args.player)

    pts_model, feature_cols = load_model("pts")
    reb_model, feature_cols2 = load_model("reb")

    if feature_cols != feature_cols2:
        raise RuntimeError("Feature lists differ between pts and reb models.")

    if pts_sigma_col not in row.index:
        raise KeyError(f"Missing column: {pts_sigma_col}")
    if reb_sigma_col not in row.index:
        raise KeyError(f"Missing column: {reb_sigma_col}")

    X = row[feature_cols].to_frame().T

    mu_pts = float(pts_model.predict(X)[0])
    mu_reb = float(reb_model.predict(X)[0])

    sigma_pts = float(row[pts_sigma_col])
    sigma_reb = float(row[reb_sigma_col])

    p_over_pts, p_under_pts = predict_player_prop(mu_pts, sigma_pts, args.pts)
    p_over_reb, p_under_reb = predict_player_prop(mu_reb, sigma_reb, args.reb)
    
    # Fair odds (no vig) from your model probabilities
    fair_over_pts = prob_to_american_odds(p_over_pts)
    fair_under_pts = prob_to_american_odds(p_under_pts)

    fair_over_reb = prob_to_american_odds(p_over_reb)
    fair_under_reb = prob_to_american_odds(p_under_reb)

    # EV at sportsbook odds (per $1 risked)
    ev_over_pts = expected_value_per_1(p_over_pts, args.over_odds)
    ev_under_pts = expected_value_per_1(p_under_pts, args.under_odds)

    ev_over_reb = expected_value_per_1(p_over_reb, args.over_odds)
    ev_under_reb = expected_value_per_1(p_under_reb, args.under_odds)
    
    
    print(f"Player: {args.player}")
    print(f"Using latest game date: {row['GAME_DATE'].date()}")
    print(f"Sigma window: {sigma_window}")

    print("\nPOINTS")
    print(f"  predicted mean (mu): {mu_pts:.2f}")
    print(f"  predicted sigma:     {sigma_pts:.2f}  (from {pts_sigma_col})")
    print(f"  line:                {args.pts}")
    print(f"  P(Over):             {p_over_pts:.3f}")
    print(f"  P(Under):            {p_under_pts:.3f}")
    print(f"  Fair odds (Over):    {fair_over_pts:+d}")
    print(f"  Fair odds (Under):   {fair_under_pts:+d}")
    print(f"  EV Over @ {args.over_odds:+.0f}:   {ev_over_pts:+.3f} per $1")
    print(f"  EV Under @ {args.under_odds:+.0f}: {ev_under_pts:+.3f} per $1")

    print("\nREBOUNDS")
    print(f"  predicted mean (mu): {mu_reb:.2f}")
    print(f"  predicted sigma:     {sigma_reb:.2f}  (from {reb_sigma_col})")
    print(f"  line:                {args.reb}")
    print(f"  P(Over):             {p_over_reb:.3f}")
    print(f"  P(Under):            {p_under_reb:.3f}")
    print(f"  Fair odds (Over):    {fair_over_reb:+d}")
    print(f"  Fair odds (Under):   {fair_under_reb:+d}")
    print(f"  EV Over @ {args.over_odds:+.0f}:   {ev_over_reb:+.3f} per $1")
    print(f"  EV Under @ {args.under_odds:+.0f}: {ev_under_reb:+.3f} per $1")




if __name__ == "__main__":
    try: 
        main()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

       

