# engine where all the core logic for prop predictions lives
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import scipy.stats as stats


@dataclass(frozen=True)

# class to hold the results of a prop prediction
class PropResult:
    # player and game information
    player_name: str
    game_date: pd.Timestamp
    sigma_window: int

    # points and rebounds lines
    pts_line: float
    reb_line: float

    # points prediction results
    mu_pts: float
    sigma_pts: float
    p_over_pts: float
    p_under_pts: float
    fair_over_pts: int
    fair_under_pts: int
    ev_over_pts: float
    ev_under_pts: float

    # rebounds prediction results
    mu_reb: float
    sigma_reb: float
    p_over_reb: float
    p_under_reb: float
    fair_over_reb: int
    fair_under_reb: int
    ev_over_reb: float
    ev_under_reb: float

# computes the cumulative distribution function (CDF) of the normal distribution and returns it as a float
def normal_cdf(z: float) -> float:
    return float(stats.norm.cdf(z))

# given a normal distribution defined by mu and sigma, and a line value,
# returns the probabilities of going over and under that line
def predict_player_prop(mu: float, sigma: float, line: float) -> tuple[float, float]:
    sigma = max(sigma, 1e-6)
    z = (line - mu) / sigma
    p_under = normal_cdf(z)
    p_over = 1.0 - p_under
    return p_over, p_under

# converts a probability (0-1) into American betting odds, assuming fair odds with no vig
# returns the odds as an integer
def prob_to_american_odds(p: float) -> int:
    """Convert probability p to American odds format (fair odds, no vig)."""
    if p <= 0.0:
        return 10_000
    if p >= 1.0:
        return -10_000
    if p >= 0.5:
        odds = -(p / (1 - p)) * 100
    else:
        odds = ((1 - p) / p) * 100
    return int(round(odds))

# converts american betting odds to the profit per 1 unit staked
# for positive odds it returns how much profit you get per 1 unit bet and for negative
# it returns how much you need to bet to win 1 unit
def american_to_profit_per_1(odds: float) -> float:
    o = odds
    if o == 0:
        raise ValueError("American odds cannot be 0")
    if o > 0:
        return o / 100.0
    return 100.0 / abs(o)

# computes the expected value per 1 unit bet given the probability of winning and the American odds
def expected_value_per_1(p_win: float, odds: float) -> float:
    profit = american_to_profit_per_1(odds)
    return p_win * profit - (1.0 - p_win)

# loads a trained model and its feature columns from disk
def load_model(model_path: Path, which: str):
    bundle = joblib.load(model_path / f"{which}_mean.joblib")
    return bundle["model"], bundle["features"]

# returns the most recent row of features for a given player name from the dataframe
def latest_features_for_player(df: pd.DataFrame, player_name: str) -> pd.Series:
    df = df.copy()
    df["PLAYER_NAME"] = df["PLAYER_NAME"].astype(str)

    subset = df[df["PLAYER_NAME"].str.lower() == player_name.lower()].copy()
    if subset.empty:
        raise ValueError(f"No features found for Player: {player_name}")

    subset = subset.sort_values("GAME_DATE", ascending=False)
    return subset.iloc[0]

# runs a full prediction pipeline for a single NBA player's points and rebounds props for an upcoming game
# returns PropResult summarizing model means, implied sigma, over/under probabilities, fair odds, and expected values
def run_prediction(
    *,
    features_path: Path,
    model_path: Path,
    player_name: str,
    pts_line: float,
    reb_line: float,
    sigma_window: int = 20,
    over_odds: float = -110,
    under_odds: float = -110,
    season: str | None = None,
) -> PropResult:
    # loads parquet file, parses GAME_DATE, and optionally filters by SEASON
    df = pd.read_parquet(features_path)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # filtering by season if provided
    if season is not None and "SEASON" in df.columns:
        df_season = df[df["SEASON"] == season].copy()
        # fallback if season not present
        if not df_season.empty:
            df = df_season

    # get latest features for player
    row = latest_features_for_player(df, player_name=player_name)

    # load models 
    pts_model, feature_cols = load_model(model_path, "pts")
    reb_model, feature_cols2 = load_model(model_path, "reb")
    # ensure feature columns match
    if feature_cols != feature_cols2:
        raise RuntimeError("Feature lists differ between pts and reb models.")

    # compute mu and sigma for points and rebounds
    pts_sigma_col = f"PTS_roll{sigma_window}_std"
    reb_sigma_col = f"REB_roll{sigma_window}_std"
    if pts_sigma_col not in row.index:
        raise KeyError(f"Missing column: {pts_sigma_col}")
    if reb_sigma_col not in row.index:
        raise KeyError(f"Missing column: {reb_sigma_col}")

    X = row[feature_cols].to_frame().T

    mu_pts = float(pts_model.predict(X)[0])
    mu_reb = float(reb_model.predict(X)[0])

    sigma_pts = float(row[pts_sigma_col])
    sigma_reb = float(row[reb_sigma_col])

    p_over_pts, p_under_pts = predict_player_prop(mu_pts, sigma_pts, pts_line)
    p_over_reb, p_under_reb = predict_player_prop(mu_reb, sigma_reb, reb_line)

    fair_over_pts = prob_to_american_odds(p_over_pts)
    fair_under_pts = prob_to_american_odds(p_under_pts)
    fair_over_reb = prob_to_american_odds(p_over_reb)
    fair_under_reb = prob_to_american_odds(p_under_reb)

    ev_over_pts = expected_value_per_1(p_over_pts, over_odds)
    ev_under_pts = expected_value_per_1(p_under_pts, under_odds)
    ev_over_reb = expected_value_per_1(p_over_reb, over_odds)
    ev_under_reb = expected_value_per_1(p_under_reb, under_odds)

    return PropResult(
        player_name=row["PLAYER_NAME"],
        game_date=row["GAME_DATE"],
        sigma_window=sigma_window,

        pts_line=pts_line,
        reb_line=reb_line,

        mu_pts=mu_pts,
        sigma_pts=sigma_pts,
        p_over_pts=p_over_pts,
        p_under_pts=p_under_pts,
        fair_over_pts=fair_over_pts,
        fair_under_pts=fair_under_pts,
        ev_over_pts=ev_over_pts,
        ev_under_pts=ev_under_pts,

        mu_reb=mu_reb,
        sigma_reb=sigma_reb,
        p_over_reb=p_over_reb,
        p_under_reb=p_under_reb,
        fair_over_reb=fair_over_reb,
        fair_under_reb=fair_under_reb,
        ev_over_reb=ev_over_reb,
        ev_under_reb=ev_under_reb,
    )
