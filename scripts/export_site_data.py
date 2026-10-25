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

Also exports every dated Transfermarkt valuation per player (for the value trend chart),
so in-season updates show between the June 30 snapshots the model uses.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import write_explanations

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
TABLE = ROOT / "data" / "processed" / "player_seasons.csv"
LIVE = ROOT / "data" / "processed" / "live_season.csv"
LIVE_STATE = ROOT / "data" / "live" / "state.json"
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
    "log_value": "value at season start", "trend_1y": "last year's value change", "trend_2y": "value change the year before",
    "age_sq": "age curve", "ga_per90": "goals + assists per 90",
    "minutes_x_value": "minutes, damped for expensive players", "goals_x_value": "goals, damped for expensive players",
    "club_league_position": "club league finish", "ucl_stage_reached": "champions league run",
    "other_league_minutes": "minutes in other leagues", "prev_minutes": "last season minutes",
    "prev_goals": "last season goals", "prev_assists": "last season assists", "promoted_club": "promoted club",
}


REASON_NAMES = {
    "age": "age", "minutes": "minutes played", "goals": "goals", "assists": "assists",
    "scoring_rate": "goals + assists per 90", "market_value_history": "market value history", "valuation_age": "how old the valuations are",
    "club_finish": "club finish", "ucl": "champions league", "promoted": "promoted club",
    "last_season": "last season", "other_league": "other leagues", "position": "position",
    "prev_league": "last season's league", "missing_history": "missing history",
}
UCL_NAMES = ["no champions league", "champions league league phase", "champions league playoff",
             "champions league last 16", "champions league quarter-final", "champions league semi-final",
             "champions league final"]
MIN_PUSH = 0.01  # ignore reasons that move the prediction by less than ~1%
MIN_CHANGE = 0.03  # predicted changes smaller than this read as "balanced"
DAMPING_FROM = 20e6  # same as train_model.py: above this, minutes and goals count for less


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


def in_progress(row):
    """True for the live season, whose stats are totals so far (the model scales them up)."""
    fraction = getattr(row, "season_fraction", np.nan)
    return pd.notna(fraction) and fraction < 1


def phrase(reason, up, row):
    """One short template phrase for a reason, given whether it pushed the value up."""
    if reason == "age":
        return f"age {int(row.age)}"  # direction comes from "pushed up" / "held back"
    so_far = " so far" if in_progress(row) else ""
    # for expensive players the model damps minutes and goals, so a good tally can still
    # pull the value down: say so instead of calling it "limited" or "only"
    if reason in ("minutes", "goals") and not up and row.value_1y_ago > DAMPING_FROM and getattr(row, reason) > 0:
        n = getattr(row, reason)
        stat = f"{n:,} minutes" if reason == "minutes" else f"{n} goal{'' if n == 1 else 's'}"
        return f"{stat}{so_far} (worth less at this price)"
    if reason == "minutes":
        return f"regular minutes ({row.minutes:,}{so_far})" if up else f"limited minutes ({row.minutes:,}{so_far})"
    if reason in ("goals", "assists"):
        n = getattr(row, reason)
        word = reason[:-1] if n == 1 else reason
        return f"{n} {word}{so_far}" if up else (f"no {reason}{so_far}" if n == 0 else f"only {n} {word}{so_far}")
    if reason == "scoring_rate":
        per90 = (row.goals + row.assists) / max(row.minutes, 90) * 90
        return f"{per90:.2f} goals + assists per 90"
    if reason == "market_value_history":
        # start value and its trend in one phrase (the model sees them as separate features)
        # the most recent year's move (3 years back only if 2 is missing), so a peak in
        # between isn't skipped (e.g. 110m -> 150m -> 100m is a fall from 150m)
        earlier = row.value_2y_ago if pd.notna(row.value_2y_ago) else row.value_3y_ago
        if pd.isna(earlier):
            return f"room to grow from {money(row.value_1y_ago)}" if up else f"already valued at {money(row.value_1y_ago)}"
        if abs(row.value_1y_ago / earlier - 1) < 0.05:
            return f"value steady at {money(row.value_1y_ago)}"
        rose = row.value_1y_ago > earlier
        # rising values that push down = the model expects a big jump to level off;
        # falling values that push up = it expects a rebound
        label = {(True, True): "rising value", (True, False): "already rose",
                 (False, False): "falling value", (False, True): "rebound after drop"}[(rose, up)]
        return f"{label} {money(earlier)} → {money(row.value_1y_ago)}"
    if reason == "missing_history":
        # describes what's actually on record: a linear model also credits *complete* history
        gaps = (["no last-season stats"] if pd.isna(row.prev_minutes) else []) + (
            ["short value history"] if pd.isna(row.value_2y_ago) or pd.isna(row.value_3y_ago) else [])
        return " and ".join(gaps) if gaps else "full history on record"
    if reason == "position":
        return f"{row.position.lower()} position"
    if reason == "club_finish":
        if pd.isna(row.club_league_position):
            return "club finish"
        place = ordinal(row.club_league_position)
        return f"club {place} so far" if in_progress(row) else f"club finished {place}"
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
    """Template sentence + per-player factor list from grouped model contributions."""
    pushes = {r: getattr(row, f"contrib_{r}") for r in reasons if pd.notna(getattr(row, f"contrib_{r}"))}
    ups = sorted([r for r, v in pushes.items() if v > MIN_PUSH], key=lambda r: -pushes[r])
    downs = sorted([r for r, v in pushes.items() if v < -MIN_PUSH], key=lambda r: pushes[r])
    change = row.predicted_value / row.value_1y_ago - 1
    text = f"predicted {money(row.predicted_value)}, {change:+.0%} from {money(row.value_1y_ago)} at season start."
    # lead with the side that matches the predicted direction, so the sentence never
    # opens with reasons that point the opposite way to the change
    rising = change >= 0
    main, other = (ups, downs) if rising else (downs, ups)
    main_label, other_label = ("pushed up by", "held back by") if rising else ("held back by", "pushed up by")
    offset = "roughly cancelled out by" if abs(change) < MIN_CHANGE else "partly offset by"
    if main:
        text += f" {main_label}: " + ", ".join(phrase(r, rising, row) for r in main[:2]) + "."
    if other:
        text += f" {offset}: " + phrase(other[0], not rising, row) + "."
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


def valuation_history(table):
    """All dated valuations per player from 3 years before the first season to the last cutoff."""
    seasons = table.season.unique()
    v = pd.read_csv(RAW / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    v = v[v.player_id.isin(table.player_id)
          & (v.date > f"{seasons.min() - 3}-06-30") & (v.date <= f"{seasons.max() + 1}-06-30")]
    return {str(pid): g.sort_values("date")[["date", "market_value_in_eur"]].values.tolist()
            for pid, g in v.groupby("player_id")}


def main():
    table = pd.read_csv(TABLE)
    if LIVE.exists():  # the season in progress, from update_live.py (brings its own cards/clean sheets)
        table = pd.concat([table, pd.read_csv(LIVE)], ignore_index=True)
        counts = ["apps", "minutes", "goals", "assists", "other_league_minutes"]
        table[counts] = table[counts].astype(int)  # concat makes them floats
    positions = pd.read_csv(RAW / "players.csv.gz", usecols=["player_id", "sub_position"])
    df = (table.merge(positions, on="player_id", how="left")
          .merge(season_events(table.season.unique()), on=["player_id", "season"], how="left", suffixes=("_live", "")))
    for col in ["yellows", "reds", "clean_sheets"]:
        if f"{col}_live" in df:
            df[col] = df[col].fillna(df.pop(f"{col}_live"))
    if PREDICTIONS.exists():
        df = df.merge(pd.read_csv(PREDICTIONS), on=["player_id", "season"], how="left")
    else:
        df["predicted_value"], df["trained_on"] = np.nan, None
    reasons = []
    if EXPLANATIONS.exists():
        contrib_rows = pd.read_csv(EXPLANATIONS).drop(columns="base")
        reasons = [c for c in contrib_rows.columns if c not in ("player_id", "season")]
        # prefix so reason names (goals, age, position...) don't collide with stat columns
        df = df.merge(contrib_rows.rename(columns={r: f"contrib_{r}" for r in reasons}), on=["player_id", "season"], how="left")
    # players with no season-start value have no price to show (and no prediction, which
    # needs a start value); drop them before ranking so ranks only count shown players
    df = df[df.value_1y_ago.notna()].copy()
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
        has_explanation = reasons and pd.notna(row.predicted_value) and pd.notna(getattr(row, f"contrib_{reasons[0]}"))
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

    live = json.loads(LIVE_STATE.read_text()) if LIVE.exists() and LIVE_STATE.exists() else None
    # Gemini rewrites (write_explanations.py), only where they still match the current facts
    ai_count = write_explanations.attach(players, live)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "seasons": sorted(df.season.unique().tolist()), "model": model_summary(), "players": players,
        "live": live,
        "valuations": valuation_history(table),
    }, ensure_ascii=False))
    print(f"wrote {len(players)} player-seasons to {OUT.relative_to(ROOT)} ({ai_count} with gemini explanations)")


if __name__ == "__main__":
    main()
