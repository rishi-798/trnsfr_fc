"""Build the training table: one row per Premier League player per season.

Reads data/raw/ (run download_data.py first) and writes data/processed/player_seasons.csv.

Rules (see project notes):
- Rows: players with at least one Premier League appearance, seasons 2022/23-2025/26.
  Seasons are labeled by start year (2025 = 2025/26).
- A season's value is the latest valuation on or before June 30 after it ends.
- History (last-season stats, earlier values) may come from any covered league.
- Missing history stays NA, with a reason column. A player who appeared for a club in a
  covered league last season but played no league minutes gets 0, not NA.
- This season's stats are Premier League only; same-season minutes in other covered
  leagues (e.g. loans out) go in separate other_league_* columns.
- value_end older than 180 days -> usable_for_training = False.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "player_seasons.csv"

SEASONS = [2022, 2023, 2024, 2025]
LEAGUE = "GB1"
HISTORY_YEARS = 3
MAX_LABEL_AGE_DAYS = 180

# Furthest Champions League stage reached. 0 = did not play in it.
UCL_STAGES = {
    "group": 1, "league phase": 1,
    "intermediate stage": 2,
    "last 16": 3,
    "quarter-finals": 4,
    "semi-finals": 5,
    "final": 6,
}


def load():
    players = pd.read_csv(RAW / "players.csv.gz", parse_dates=["date_of_birth"])
    valuations = pd.read_csv(RAW / "player_valuations.csv.gz", parse_dates=["date"])
    appearances = pd.read_csv(RAW / "appearances.csv.gz")
    games = pd.read_csv(RAW / "games.csv.gz")
    competitions = pd.read_csv(RAW / "competitions.csv.gz")
    return players, valuations, appearances, games, competitions


def season_cutoff(season, years_back=0):
    """June 30 after the season ends, shifted back by years_back seasons."""
    return pd.to_datetime((season + 1 - years_back).astype(str) + "-06-30")


def value_at(rows, valuations, cutoff_col, prefix):
    """Attach the latest valuation on or before each row's cutoff date."""
    matched = pd.merge_asof(
        rows.sort_values(cutoff_col),
        valuations[["player_id", "date", "market_value_in_eur"]].sort_values("date"),
        left_on=cutoff_col, right_on="date", by="player_id", direction="backward",
    )
    rows = matched.rename(columns={"market_value_in_eur": prefix})
    rows[f"{prefix}_age_days"] = (rows[cutoff_col] - rows["date"]).dt.days
    return rows.drop(columns=["date", cutoff_col])


def ucl_stage(round_name):
    name = str(round_name).lower()
    if name.startswith("group"):
        return 1
    for key, stage in sorted(UCL_STAGES.items(), key=lambda kv: -kv[1]):
        if name.startswith(key):
            return stage
    return 0


def attach_previous_season(rows, apps, games, covered_leagues):
    """Last season's league stats (any covered league) for each (player_id, season) row.

    prev_league = where they played the most minutes. A player who appeared (cups, Europe)
    for a club in a covered league but played no league minutes gets 0, not NA.
    """
    league_apps = apps[apps.competition_id.isin(covered_leagues)]
    prev = league_apps.groupby(["player_id", "season", "competition_id"]).agg(
        minutes=("minutes_played", "sum"), goals=("goals", "sum"), assists=("assists", "sum")
    ).reset_index()
    prev_league = (
        prev.sort_values("minutes", ascending=False)
        .drop_duplicates(["player_id", "season"])[["player_id", "season", "competition_id"]]
        .rename(columns={"competition_id": "prev_league"})
    )
    prev_totals = prev.groupby(["player_id", "season"])[["minutes", "goals", "assists"]].sum().reset_index()
    prev_stats = prev_totals.merge(prev_league, on=["player_id", "season"])
    prev_stats["season"] += 1  # align last season's stats with this season's row
    prev_stats = prev_stats.rename(columns={"minutes": "prev_minutes", "goals": "prev_goals", "assists": "prev_assists"})
    rows = rows.merge(prev_stats, on=["player_id", "season"], how="left")

    # club = the one they appeared for most often last season
    league_games = games[games.competition_id.isin(covered_leagues)]
    club_league = pd.concat([
        league_games[["home_club_id", "season", "competition_id"]].set_axis(["club_id", "season", "league"], axis=1),
        league_games[["away_club_id", "season", "competition_id"]].set_axis(["club_id", "season", "league"], axis=1),
    ]).drop_duplicates(["club_id", "season"])
    last_club = (
        apps.groupby(["player_id", "season", "player_club_id"]).size().rename("n").reset_index()
        .sort_values("n", ascending=False).drop_duplicates(["player_id", "season"])
        .rename(columns={"player_club_id": "club_id"})
        .merge(club_league, on=["club_id", "season"])[["player_id", "season", "league"]]
    )
    last_club["season"] += 1
    rows = rows.merge(last_club, on=["player_id", "season"], how="left")
    true_zero = rows.prev_minutes.isna() & rows.league.notna()
    rows.loc[true_zero, ["prev_minutes", "prev_goals", "prev_assists"]] = 0
    rows.loc[true_zero, "prev_league"] = rows.loc[true_zero, "league"]
    rows = rows.drop(columns="league")
    return rows


def attach_values(rows, valuations):
    """The answer (value at the end of the season) and the history (1-3 seasons before),
    plus why last season's stats are missing (checked after values are attached)."""
    rows["cutoff"] = season_cutoff(rows.season)
    rows = value_at(rows, valuations, "cutoff", "value_end")
    for years in range(1, HISTORY_YEARS + 1):
        rows["cutoff"] = season_cutoff(rows.season, years_back=years)
        rows = value_at(rows, valuations, "cutoff", f"value_{years}y_ago")
        rows[f"value_{years}y_ago_missing_reason"] = rows[f"value_{years}y_ago"].isna().map({True: "no_valuation", False: None})

    no_prev = rows.prev_minutes.isna()
    rows["prev_stats_missing_reason"] = None
    rows.loc[no_prev & rows.value_1y_ago.notna(), "prev_stats_missing_reason"] = "no_stats_in_dataset"
    rows.loc[no_prev & rows.value_1y_ago.isna(), "prev_stats_missing_reason"] = "no_valuation"
    return rows


def main():
    players, valuations, appearances, games, competitions = load()
    clubs = pd.read_csv(RAW / "clubs.csv.gz", usecols=["club_id", "name"]).rename(columns={"name": "club_name"})
    covered_leagues = set(competitions.loc[competitions.type == "domestic_league", "competition_id"])

    apps = appearances.merge(games[["game_id", "season"]], on="game_id")
    league_apps = apps[apps.competition_id.isin(covered_leagues)]
    pl_apps = league_apps[(league_apps.competition_id == LEAGUE) & league_apps.season.isin(SEASONS)]

    # This season's Premier League stats
    rows = (
        pl_apps.groupby(["player_id", "season"])
        .agg(apps=("game_id", "size"), minutes=("minutes_played", "sum"),
             goals=("goals", "sum"), assists=("assists", "sum"))
        .reset_index()
    )

    # Club = the club the player had the most PL minutes for that season (handles mid-season movers)
    club_minutes = pl_apps.groupby(["player_id", "season", "player_club_id"]).minutes_played.sum().reset_index()
    main_club = (
        club_minutes.sort_values("minutes_played", ascending=False)
        .drop_duplicates(["player_id", "season"])
        .rename(columns={"player_club_id": "club_id"})[["player_id", "season", "club_id"]]
    )
    rows = rows.merge(main_club, on=["player_id", "season"]).merge(clubs, on="club_id", how="left")

    # Final league position: the club's position after its last league game of the season
    pl_games = games[games.competition_id == LEAGUE]
    positions = pd.concat([
        pl_games[["season", "date", "home_club_id", "home_club_position"]]
        .set_axis(["season", "date", "club_id", "club_league_position"], axis=1),
        pl_games[["season", "date", "away_club_id", "away_club_position"]]
        .set_axis(["season", "date", "club_id", "club_league_position"], axis=1),
    ])
    final_positions = positions.sort_values("date").drop_duplicates(["season", "club_id"], keep="last")
    rows = rows.merge(final_positions[["season", "club_id", "club_league_position"]], on=["season", "club_id"], how="left")

    # Was the club in the Premier League last season? (False = promoted)
    pl_club_seasons = set(zip(positions.club_id, positions.season))
    rows["club_in_pl_last_season"] = [(c, s - 1) in pl_club_seasons for c, s in zip(rows.club_id, rows.season)]

    # Champions League: furthest stage the club reached
    ucl = games[games.competition_id == "CL"].copy()
    ucl["stage"] = ucl["round"].map(ucl_stage)
    ucl_clubs = pd.concat([
        ucl[["season", "home_club_id", "stage"]].set_axis(["season", "club_id", "stage"], axis=1),
        ucl[["season", "away_club_id", "stage"]].set_axis(["season", "club_id", "stage"], axis=1),
    ])
    furthest = ucl_clubs.groupby(["season", "club_id"]).stage.max().rename("ucl_stage_reached").reset_index()
    rows = rows.merge(furthest, on=["season", "club_id"], how="left")
    rows["ucl_stage_reached"] = rows.ucl_stage_reached.fillna(0).astype(int)

    # Age on 1 August of the season, and position
    rows = rows.merge(players[["player_id", "name", "date_of_birth", "position"]], on="player_id", how="left")
    season_start = pd.to_datetime(rows.season.astype(str) + "-08-01")
    rows["age"] = ((season_start - rows.date_of_birth).dt.days / 365.25).round(1)

    rows = attach_previous_season(rows, apps, games, covered_leagues)

    # Same-season stats in other covered leagues (e.g. loaned out mid-season); 0 if none
    other = (
        league_apps[(league_apps.competition_id != LEAGUE) & league_apps.season.isin(SEASONS)]
        .groupby(["player_id", "season"])
        .agg(other_league_minutes=("minutes_played", "sum"), other_league_goals=("goals", "sum"),
             other_league_assists=("assists", "sum"))
        .reset_index()
    )
    rows = rows.merge(other, on=["player_id", "season"], how="left")
    other_columns = ["other_league_minutes", "other_league_goals", "other_league_assists"]
    rows[other_columns] = rows[other_columns].fillna(0).astype(int)

    rows = attach_values(rows, valuations)

    rows["usable_for_training"] = rows.value_end.notna() & (rows.value_end_age_days <= MAX_LABEL_AGE_DAYS)

    columns = [
        "player_id", "name", "season", "club_id", "club_name",  # identifiers, not features
        "age", "position", "apps", "minutes", "goals", "assists",
        "other_league_minutes", "other_league_goals", "other_league_assists",
        "club_league_position", "ucl_stage_reached", "club_in_pl_last_season",
        "prev_league", "prev_minutes", "prev_goals", "prev_assists", "prev_stats_missing_reason",
    ]
    for years in range(1, HISTORY_YEARS + 1):
        columns += [f"value_{years}y_ago", f"value_{years}y_ago_age_days", f"value_{years}y_ago_missing_reason"]
    columns += ["value_end", "value_end_age_days", "usable_for_training"]

    table = rows[columns].sort_values(["season", "player_id"]).reset_index(drop=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(f"wrote {len(table)} rows to {OUT.relative_to(ROOT)}")
    print(f"usable for training: {table.usable_for_training.sum()}")


if __name__ == "__main__":
    main()
