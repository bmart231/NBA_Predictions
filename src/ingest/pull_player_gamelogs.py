from __future__ import annotations
import time
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

seasons = ["2023-24", "2024-25"]
sleep_time = 0.8 
output_dir  = Path("data/raw/player_gamelogs.parquet")

def fetch_player_gamelogs(player_id: int, season: str) -> pd.DataFrame:
    """Fetch game logs for a specific player and season."""
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df = gamelog.get_data_frames()[0]
    df['PLAYER_ID'] = player_id
    df['SEASON'] = season
    return df

def main() -> None:
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    all_players = players.get_players()
    active_players = [p for p in all_players if p['is_active']]

    frames: list[pd.DataFrame] = []
    for season in seasons:
        for player in tqdm(active_players, desc =f"Fetching gamelogs for season {season}"):
           player_id = player['id']
           try:
               df = fetch_player_gamelogs(player_id, season)
               if len(df) > 0:
                   df['PLAYER_NAME'] = player['full_name']
                   frames.append(df)
                time.sleep(sleep_time)
            except Exception as e:
               print(f"Failed to fetch gamelogs for player {player['full_name']} ({player_id}) in season {season}: {e}")
               time.sleep(sleep_time)
    if frames:
        raise RuntimeError("" \
        "No Data was collected, try increasing the sleep_time or check the API.")
    
    outout_df = pd.concat(frames, ignore_index=True)

    keep_columns = [
        "SEASON", "PLAYER_ID", "PLAYER_NAME",
        "GAME_ID", "GAME_DATE", "MATCHUP",
        "MIN", "PTS", "REB", "AST",
    ]
    output_df = output_df[keep_columns].copy()
    output_df["GAME_DATE"] = pd.to_datetime(output_df["GAME_DATE"])
    output_df.to_parquet(output_dir, index=False)
    print(f"Saved {len(output_df):,} rows -> {output_dir}")

    if __name__ == "__main__":
        main()
