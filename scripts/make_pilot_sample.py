"""Draw the 50-player pilot sample for manual data-quality review.

Reads data/processed/player_seasons.csv (run build_training_table.py first) and writes
data/processed/pilot_sample.csv with a category and empty review columns to fill in
by comparing each row against the player's Transfermarkt page.

Categories are filled rarest first; each row and each player is used at most once.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TABLE = ROOT / "data" / "processed" / "player_seasons.csv"
OUT = ROOT / "data" / "processed" / "pilot_sample.csv"

SEED = 42
LIMITED_MINUTES = 450
MAX_LABEL_AGE_DAYS = 180

# (category, count), in the order they are filled: rarest first, complete records last
STRATA = [
    ("missing_end_value", 5),
    ("stale_end_value", 5),
    ("mid_season_mover", 7),
    ("limited_minutes", 5),
    ("promoted_club", 8),
    ("missing_prev_stats", 10),
    ("complete", 10),
]


def mid_season_movers():
    """(player_id, season) pairs where the player played for more than one PL club."""
    appearances = pd.read_csv(RAW / "appearances.csv.gz", usecols=["game_id", "player_id", "player_club_id", "competition_id"])
    games = pd.read_csv(RAW / "games.csv.gz", usecols=["game_id", "season"])
    pl = appearances[appearances.competition_id == "GB1"].merge(games, on="game_id")
    clubs = pl.groupby(["player_id", "season"]).player_club_id.nunique()
    return set(clubs[clubs > 1].index)


def has_review_entries(path):
    if not path.exists():
        return False
    existing = pd.read_csv(path)
    return existing[["review_status", "review_notes"]].notna().any().any()


def main():
    # Never overwrite a sample someone has started reviewing
    if has_review_entries(OUT) and "--force" not in sys.argv:
        sys.exit(f"{OUT.relative_to(ROOT)} has review entries; not overwriting. "
                 "Move or rename it first, or rerun with --force to discard the review.")

    table = pd.read_csv(TABLE)
    movers = mid_season_movers()
    data_columns = [c for c in table.columns if not c.endswith("_missing_reason")]

    checks = {
        "missing_end_value": table.value_end.isna(),
        "stale_end_value": table.value_end_age_days > MAX_LABEL_AGE_DAYS,
        "mid_season_mover": pd.Series([(p, s) in movers for p, s in zip(table.player_id, table.season)]),
        "limited_minutes": table.minutes < LIMITED_MINUTES,
        "promoted_club": ~table.club_in_pl_last_season,
        "missing_prev_stats": table.prev_minutes.isna(),
        "complete": table[data_columns].notna().all(axis=1) & table.usable_for_training,
    }

    picked = []
    used_players = set()
    for category, count in STRATA:
        pool = table[checks[category] & ~table.player_id.isin(used_players)]
        pool = pool.drop_duplicates("player_id")  # one season per player within a category
        sample = pool.sample(n=min(count, len(pool)), random_state=SEED).assign(category=category)
        if len(sample) < count:
            print(f"warning: only {len(sample)} available for {category} (wanted {count})")
        picked.append(sample)
        used_players.update(sample.player_id)

    pilot = pd.concat(picked)
    pilot["review_status"] = ""  # e.g. ok / issue
    pilot["review_notes"] = ""
    pilot = pilot[["category", "review_status", "review_notes"] + list(table.columns)]
    pilot.to_csv(OUT, index=False)
    print(f"wrote {len(pilot)} rows to {OUT.relative_to(ROOT)}")
    print(pilot.category.value_counts().to_string())


if __name__ == "__main__":
    main()
