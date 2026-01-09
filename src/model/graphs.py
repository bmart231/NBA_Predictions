import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]   # NBA_Predictions/
DATA_PATH = repo_root / "data" / "raw" / "player_gamelogs.parquet"

SEASON = "2024-25"
MIN_GAMES = 20
TOP_N = 10


def main():
    df = pd.read_parquet(DATA_PATH)

    if "SEASON" in df.columns:
        df = df[df["SEASON"] == SEASON].copy()

    summary = (
        df.groupby(["PLAYER_ID", "PLAYER_NAME"])
          .agg(games=("GAME_DATE", "count"), ppg=("PTS", "mean"))
          .reset_index()
    )

    summary = summary[summary["games"] >= MIN_GAMES]
    top = summary.sort_values("ppg", ascending=False).head(TOP_N)

    plt.figure(figsize=(10, 5))
    plt.bar(top["PLAYER_NAME"], top["ppg"])
    plt.title(f"Top {TOP_N} Players by PPG ({SEASON}, min {MIN_GAMES} games)")
    plt.xlabel("Player")
    plt.ylabel("Points per Game")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # ensure graphs/ exists relative to repo root
    out_dir = repo_root / "graphs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / "top10_ppg.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
