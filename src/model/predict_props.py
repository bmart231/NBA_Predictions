# Take a player + matchup (home/away) and predict over/under prop for that player
from __future__ import annotations
from pathlib import Path
import sys

from src.model.prop_engine import run_prediction

import argparse

features_path = Path("data/processed/player_features_2023_2026.parquet") # path to processed features
model_path = Path("models") # path to trained models


def argparse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict NBA player prop over/under probabilities.") # description of the script
    parser.add_argument("--player", help='Player name, e.g. "LeBron James"') # player name argument
    parser.add_argument("--pts", type=float, help="Points line, e.g. 27.5") # points line argument
    parser.add_argument("--reb", type=float, help="Rebounds line, e.g. 7.5") # rebounds line argument
    parser.add_argument("--over-odds", type=float, default=None, help="Sportsbook odds for OVER (e.g., -110)") # over odds argument
    parser.add_argument("--under-odds", type=float, default=None, help="Sportsbook odds for UNDER (e.g., -110)") # under odds argument
    
    # Add these two missing arguments:
    parser.add_argument("--sigma-window", type=int, default=10, help="Window size for sigma calculation (default: 10)")
    parser.add_argument("--season", type=str, default=None, help="Season filter, e.g. '2024-25'")

    return parser.parse_args()

def main() -> None:
    args = argparse_args()

    if not args.player:
        args.player = input("Player name (e.g., LeBron James): ").strip()

    if args.pts is None:
        args.pts = float(input("Points line (e.g., 27.5): ").strip())

    if args.reb is None:
        args.reb = float(input("Rebounds line (e.g., 7.5): ").strip())

    if args.over_odds is None:
        args.over_odds = float(input("Over odds (American, e.g., -110 or +150): ").strip())

    if args.under_odds is None:
        args.under_odds = float(input("Under odds (American, e.g., -110 or +150): ").strip())

    # run prediction
    res = run_prediction(
        
        features_path=features_path,
        model_path=model_path,
        player_name=args.player,
        pts_line=args.pts,
        reb_line=args.reb,
        sigma_window=args.sigma_window,
        over_odds=args.over_odds,
        under_odds=args.under_odds,
        season=args.season,
    )

    
    
    print(f"Player: {res.player_name}")
    print(f"Using latest game date: {res.game_date.date()}")
    if args.season:
        print(f"Season filter: {args.season}")
    print(f"Sigma window: {res.sigma_window}")

    print("\nPOINTS")
    print(f"  mu: {res.mu_pts:.2f}   sigma: {res.sigma_pts:.2f}   line: {res.pts_line}")
    print(f"  P(Over): {res.p_over_pts:.3f}   P(Under): {res.p_under_pts:.3f}")
    print(f"  Fair odds Over: {res.fair_over_pts:+d}   Under: {res.fair_under_pts:+d}")
    print(f"  EV Over @ {args.over_odds:+.0f}: {res.ev_over_pts:+.3f} per $1")
    print(f"  EV Under @ {args.under_odds:+.0f}: {res.ev_under_pts:+.3f} per $1")

    print("\nREBOUNDS")
    print(f"  mu: {res.mu_reb:.2f}   sigma: {res.sigma_reb:.2f}   line: {res.reb_line}")
    print(f"  P(Over): {res.p_over_reb:.3f}   P(Under): {res.p_under_reb:.3f}")
    print(f"  Fair odds Over: {res.fair_over_reb:+d}   Under: {res.fair_under_reb:+d}")
    print(f"  EV Over @ {args.over_odds:+.0f}: {res.ev_over_reb:+.3f} per $1")
    print(f"  EV Under @ {args.under_odds:+.0f}: {res.ev_under_reb:+.3f} per $1")

# Run the prediction process
if __name__ == "__main__":
    try: 
        main()
    except ValueError as e: # catch value errors (e.g., player not found)
        print(f"Error: {e}")
        sys.exit(1)

       

