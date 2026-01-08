from __future__ import annotations

from pathlib import Path
import pandas as pd

raw_gamelogs_path = Path("data/raw/player_gamelogs.parquet")
processed_features_path = Path("data/processed/player_features.parquet")

rolling_window = [5, 10, 20]  # rolling averages over last 5, 10, 20 games

# this function adds two binary features indicating whether the game was played at home or away.
# it returs a copy of the original dataframe with the new features added.
def add_home_away_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features indicating if the game was home or away."""
    df = df.copy()
    df["IS_HOME"] = df["MATCHUP"].str.contains(" vs. ").astype(int)
    df["IS_AWAY"] = df["MATCHUP"].str.contains(" @ ").astype(int)
    return df

# This function computes how many days of rest a player had before each game and stores it in a new rest_day col. 
# Rest days are capped to the range 0-10 to avoid extreme values. and the functions returns a sorted copy of the 
# original dataframe with the new feature added.
def add_rest_days_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a feature for rest days before each game."""
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    prev_date = df.groupby("PLAYER_ID")["GAME_DATE"].shift(1)
    df["rest_days"] = (df["GAME_DATE"] - prev_date).dt.days # type: ignore # should still work regardless
    df["rest_days"] = df["rest_days"].clip(lower=0, upper=10)
    return df

# this functions adds rolling mean and std features for a given column (like MIN, PTS, REB).
# computed seperately for each player and shifted by 1 to avoid data leakage.
def add_roll_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Add rolling average features for a given column."""
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).copy()
    game_stats = df.groupby("PLAYER_ID", group_keys=False)[col]

    for window in rolling_window:
        df[f"{col}_roll{window}_mean"] = game_stats.apply(
            lambda s: s.shift(1).rolling(window, min_periods=3).mean()
        )
        df[f"{col}_roll{window}_std"] = game_stats.apply(
            lambda s: s.shift(1).rolling(window, min_periods=3).std()
        )
    return df

# main function builds a feature-enriched datasetof NBA player gamelogs and saves it as a parquet file.
# it cleans raw gamelog data, derives target variables (y_pts, y_reb), and adds features like home/away indicators,
# rest days, and rolling statistics for minutes, points, and rebounds.
def main() -> None:
    processed_features_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(raw_gamelogs_path)

    # basic cleaning and type conversions
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # convert relevant columns to numeric types (only columns that exist)
    for col in ["MIN", "PTS", "AST", "REB"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # drop rows with missing essential data
    df = df.dropna(subset=["PLAYER_ID", "GAME_DATE", "MIN", "PTS", "REB"]).copy()

    # targets (actual outcomes)
    df["y_pts"] = df["PTS"].astype(float)
    df["y_reb"] = df["REB"].astype(float)

    # feature additions
    df = add_home_away_features(df)
    df = add_rest_days_feature(df)

    df = add_roll_features(df, "MIN")
    df = add_roll_features(df, "PTS")
    df = add_roll_features(df, "REB")

    feature_cols = [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")] + [
        "rest_days",
        "IS_HOME",
        "IS_AWAY",
    ]

    df = df.dropna(subset=feature_cols).copy()

    keep = [
        "SEASON", "PLAYER_ID", "PLAYER_NAME", "GAME_DATE", "MATCHUP",
        "MIN", "PTS", "REB", "AST",
        "rest_days", "IS_HOME", "IS_AWAY",
        "y_pts", "y_reb",
    ] + [c for c in df.columns if c.endswith("_mean") or c.endswith("_std")]

    df = df[keep].copy()

    df.to_parquet(processed_features_path, index=False)
    print(f"Saved {len(df):,} rows -> {processed_features_path}")

# Run the feature building process
if __name__ == "__main__":
    main()
