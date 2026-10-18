"""Export the data the site needs to site/data/players.json.

Reads data/processed/player_seasons.csv (run build_training_table.py first) plus raw
appearances/games for cards and clean sheets, and computes the performance score.

Performance score (per 90 minutes, Premier League only, simplified to available stats):
  forwards:    5*goals + 4*assists               - 2*yellows - 5*reds
  midfielders: 4*assists + 3*goals               - 2*yellows - 5*reds
  defenders:   3*clean_sheets + 3*goals + 3*assists - 2*yellows - 5*reds
  goalkeepers: 5*clean_sheets                     - 2*yellows - 5*reds
then multiplied by minutes / (minutes + 900) so small samples don't dominate.
A clean sheet = played 60+ minutes and the team conceded 0.
score_pct = percentile of the score within the same season and position (0-100).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TABLE = ROOT / "data" / "processed" / "player_seasons.csv"
PREDICTIONS = ROOT / "data" / "processed" / "predictions.csv"
EXPLANATIONS = ROOT / "data" / "processed" / "explanations.csv"
METRICS = ROOT / "data" / "processed" / "model_metrics.json"
OUT = ROOT / "site" / "data" / "players.json"

LEAGUE = "GB1"
RELIABILITY_MINUTES = 900
CLEAN_SHEET_MIN_MINUTES = 60

INITIALS = {
    "Centre-Forward": "st", "Second Striker": "ss", "Left Winger": "lw", "Right Winger": "rw",
    "Attacking Midfield": "cam", "Central Midfield": "cm", "Defensive Midfield": "cdm",
    "Left Midfield": "lm", "Right Midfield": "rm",
    "Centre-Back": "cb", "Left-Back": "lb", "Right-Back": "rb", "Goalkeeper": "gk",
}
GROUPS = {"Attack": "fwd", "Midfield": "mid", "Defender": "def", "Goalkeeper": "gk"}


def season_events(seasons):
    """Yellow cards, red cards and clean sheets per player-season in the Premier League."""
    apps = pd.read_csv(RAW / "appearances.csv.gz", usecols=[
        "game_id", "player_id", "player_club_id", "competition_id", "yellow_cards", "red_cards", "minutes_played"])
    games = pd.read_csv(RAW / "games.csv.gz", usecols=[
        "game_id", "season", "home_club_id", "home_club_goals", "away_club_goals"])
    apps = apps[apps.competition_id == LEAGUE].merge(games, on="game_id")
    apps = apps[apps.season.isin(seasons)]

    at_home = apps.player_club_id == apps.home_club_id
    conceded = np.where(at_home, apps.away_club_goals, apps.home_club_goals)
    apps["clean_sheet"] = ((conceded == 0) & (apps.minutes_played >= CLEAN_SHEET_MIN_MINUTES)).astype(int)

    return apps.groupby(["player_id", "season"]).agg(
        yellows=("yellow_cards", "sum"), reds=("red_cards", "sum"), clean_sheets=("clean_sheet", "sum"),
    ).reset_index()


def performance_score(df):
    per90 = lambda col: df[col] / df.minutes * 90
    g, a, cs, y, r = per90("goals"), per90("assists"), per90("clean_sheets"), per90("yellows"), per90("reds")
    raw = pd.Series(np.select(
        [df.group == "fwd", df.group == "mid", df.group == "def", df.group == "gk"],
        [5 * g + 4 * a, 4 * a + 3 * g, 3 * cs + 3 * g + 3 * a, 5 * cs],
        default=np.nan,
    ), index=df.index) - 2 * y - 5 * r
    return raw * df.minutes / (df.minutes + RELIABILITY_MINUTES)


def short_club(name):
    for suffix in (" AFC", " FC"):
        if isinstance(name, str) and name.endswith(suffix):
            return name[: -len(suffix)]
    return name


FEATURE_LABELS = {
    "age": "age", "minutes": "minutes played", "goals": "goals", "assists": "assists",
    "value_1y_ago": "value at season start", "value_2y_ago": "value 2 years ago", "value_3y_ago": "value 3 years ago",
    "club_league_position": "club league finish", "ucl_stage_reached": "champions league run",
    "other_league_minutes": "minutes in other leagues", "prev_minutes": "last season minutes",
    "prev_goals": "last season goals", "prev_assists": "last season assists", "promoted_club": "promoted club",
}


REASON_NAMES = {
    "age": "age", "minutes": "minutes played", "goals": "goals", "assists": "assists",
    "start_value": "value at season start", "value_history": "value history", "valuation_age": "how old the valuations are",
    "club_finish": "club finish", "ucl": "champions league", "promoted": "promoted club",
    "last_season": "last season", "other_league": "other leagues", "position": "position",
    "prev_league": "last season's league", "missing_history": "missing history",
}
UCL_NAMES = ["no champions league", "champions league league phase", "champions league playoff",
             "champions league last 16", "champions league quarter-final", "champions league semi-final",
             "champions league final"]
MIN_PUSH = 0.01  # ignore reasons that move the prediction by less than ~1%


def money(v):
    if v is None or pd.isna(v):
        return "?"
    if v >= 1e6:
        m = v / 1e6
        return f"€{round(m)}m" if m >= 10 else f"€{m:.1f}".rstrip("0").rstrip(".") + "m"
    return f"€{round(v / 1e3)}k"


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def phrase(reason, up, row):
    """One short template phrase for a reason, given whether it pushed the value up."""
    if reason == "age":
        return f"age {int(row.age)}"  # direction comes from "pushed up" / "held back"
    if reason == "minutes":
        return f"regular minutes ({row.minutes:,})" if up else f"limited minutes ({row.minutes:,})"
    if reason in ("goals", "assists"):
        n = getattr(row, reason)
        return f"{n} {reason}" if up else (f"no {reason}" if n == 0 else f"only {n} {reason}")
    if reason == "start_value":
        return f"room to grow from {money(row.value_1y_ago)}" if up else f"already valued at {money(row.value_1y_ago)}"
    if reason == "value_history":
        return f"value history ({money(row.value_3y_ago)} → {money(row.value_1y_ago)})"
    if reason == "club_finish":
        return f"club finished {ordinal(row.club_league_position)}" if pd.notna(row.club_league_position) else "club finish"
    if reason == "ucl":
        return UCL_NAMES[int(row.ucl_stage_reached)]
    if reason == "promoted":
        return "promoted club" if not row.club_in_pl_last_season else "established pl club"
    if reason == "last_season":
        if pd.isna(row.prev_minutes):
            return "no last-season data"
        return f"last season ({int(row.prev_minutes):,} min, {int(row.prev_goals)}g {int(row.prev_assists)}a)"
    if reason == "other_league":
        return f"{row.other_league_minutes:,} min in other leagues" if row.other_league_minutes else "no minutes elsewhere"
    return REASON_NAMES[reason]


def explanation(row, reasons):
    """Template sentence + per-player factor list from grouped SHAP contributions."""
    pushes = {r: getattr(row, f"shap_{r}") for r in reasons if pd.notna(getattr(row, f"shap_{r}"))}
    ups = sorted([r for r, v in pushes.items() if v > MIN_PUSH], key=lambda r: -pushes[r])[:2]
    downs = sorted([r for r, v in pushes.items() if v < -MIN_PUSH], key=lambda r: pushes[r])[:1]
    change = row.predicted_value / row.value_1y_ago - 1
    text = f"{money(row.predicted_value)} ({change:+.0%} from {money(row.value_1y_ago)})."
    if ups:
        text += " pushed up by: " + ", ".join(phrase(r, True, row) for r in ups) + "."
    if downs:
        text += " held back by: " + ", ".join(phrase(r, False, row) for r in downs) + "."
    # effect of each reason on its own, as a % change (contributions multiply in euro terms)
    factors = sorted(
        ({"name": REASON_NAMES[r], "effect": round(float(np.exp(v) - 1), 3)} for r, v in pushes.items() if abs(v) > MIN_PUSH),
        key=lambda f: -abs(f["effect"]),
    )[:6]
    return text, factors


def model_summary():
    """Test-season scores and the top global factors, if train_model.py has been run."""
    if not METRICS.exists():
        return None
    metrics = json.loads(METRICS.read_text())
    top = [(k, v) for k, v in metrics["importances"].items() if v > 0][:8]
    return {
        "test_season": metrics["test_season"],
        "results": metrics["results"]["all"],
        "factors": [{"name": FEATURE_LABELS.get(k, k.replace("_", " ")), "weight": v} for k, v in top],
    }


def main():
    table = pd.read_csv(TABLE)
    positions = pd.read_csv(RAW / "players.csv.gz", usecols=["player_id", "sub_position"])
    df = (table.merge(positions, on="player_id", how="left")
          .merge(season_events(table.season.unique()), on=["player_id", "season"], how="left"))
    if PREDICTIONS.exists():
        df = df.merge(pd.read_csv(PREDICTIONS), on=["player_id", "season"], how="left")
    else:
        df["predicted_value"], df["trained_on"] = np.nan, None
    reasons = []
    if EXPLANATIONS.exists():
        shap_rows = pd.read_csv(EXPLANATIONS).drop(columns="base")
        reasons = [c for c in shap_rows.columns if c not in ("player_id", "season")]
        # prefix so reason names (goals, age, position...) don't collide with stat columns
        df = df.merge(shap_rows.rename(columns={r: f"shap_{r}" for r in reasons}), on=["player_id", "season"], how="left")
    df[["yellows", "reds", "clean_sheets"]] = df[["yellows", "reds", "clean_sheets"]].fillna(0).astype(int)

    df["group"] = df.position.map(GROUPS)
    df["initials"] = df.sub_position.map(INITIALS).fillna(df.group)
    df["score"] = performance_score(df).round(2)
    df["score_pct"] = (df.groupby(["season", "group"]).score.rank(pct=True) * 100).round()
    # rank within position and season, e.g. 1 of 182 forwards (1 = best)
    df["score_rank"] = df.groupby(["season", "group"]).score.rank(ascending=False, method="min")
    df["group_size"] = df.groupby(["season", "group"]).score.transform("count")

    players = []
    for row in df.itertuples():
        clean = lambda v: None if pd.isna(v) else (int(v) if float(v).is_integer() else float(v))
        has_explanation = reasons and pd.notna(row.predicted_value) and pd.notna(getattr(row, f"shap_{reasons[0]}"))
        explain_text, factors = explanation(row, reasons) if has_explanation else (None, [])
        players.append({
            "id": int(row.player_id), "name": row.name, "season": int(row.season),
            "club": short_club(row.club_name), "group": row.group, "initials": row.initials,
            "age": clean(row.age), "apps": int(row.apps), "minutes": int(row.minutes),
            "goals": int(row.goals), "assists": int(row.assists), "clean_sheets": int(row.clean_sheets),
            "yellows": int(row.yellows), "reds": int(row.reds),
            "other_league_minutes": int(row.other_league_minutes),
            "league_position": clean(row.club_league_position), "ucl_stage": int(row.ucl_stage_reached),
            "promoted": not row.club_in_pl_last_season,
            "prev": {"league": None if pd.isna(row.prev_league) else row.prev_league,
                     "minutes": clean(row.prev_minutes), "goals": clean(row.prev_goals),
                     "assists": clean(row.prev_assists)},
            # oldest -> newest: 3y ago, 2y ago, 1y ago (start of season), end of season
            "values": [clean(row.value_3y_ago), clean(row.value_2y_ago), clean(row.value_1y_ago), clean(row.value_end)],
            "value_end_age_days": clean(row.value_end_age_days),
            "usable_for_training": bool(row.usable_for_training),
            "score": clean(row.score), "score_pct": clean(row.score_pct),
            "score_rank": clean(row.score_rank), "group_size": int(row.group_size),
            "predicted": clean(row.predicted_value),
            "trained_on": None if pd.isna(row.trained_on) else row.trained_on,
            "explanation": explain_text, "factors": factors,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seasons": sorted(df.season.unique().tolist()), "model": model_summary(), "players": players,
    }, ensure_ascii=False))
    print(f"wrote {len(players)} player-seasons to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
