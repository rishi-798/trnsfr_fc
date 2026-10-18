"""Build the live 2026/27 season rows from the Fantasy Premier League (FPL) feed.

The Transfermarkt dataset stops at the end of 2025/26, but everything the model needs
from before the season (season-start value on 30 June 2026, value history, last
season's stats) is already in it. This script adds the season in progress from FPL:
squads, positions, minutes, goals, assists, cards, clean sheets, and the live table.

Only rebuilds when FPL marks a new gameweek as final (finished and data checked), so
running it daily is cheap. Partial seasons are scaled up by the model (see
train_model.features), using season_fraction = club games played / 38.

Players are matched to Transfermarkt by birth date + shared name words; then by close
name + birth date within 2 days (or day/month swapped); then by data/live/fpl_matches.csv
(manual: fpl_code -> player_id). Unmatched players can't be priced and are left out.

Usage:
  python scripts/update_live.py           build if there's a new final gameweek
  python scripts/update_live.py --check   only report whether there is one (for CI)
  python scripts/update_live.py --force   rebuild anyway

Writes data/processed/live_season.csv, data/live/matches.csv (for review) and
data/live/state.json.
"""
import difflib
import json
import os
import re
import ssl
import sys
import unicodedata
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi
import pandas as pd

import build_training_table as btt

ROOT = Path(__file__).resolve().parent.parent
LIVE_DIR = ROOT / "data" / "live"
STATE = LIVE_DIR / "state.json"
MATCHES = LIVE_DIR / "matches.csv"
MANUAL_MATCHES = LIVE_DIR / "fpl_matches.csv"
OUT = ROOT / "data" / "processed" / "live_season.csv"

FPL = "https://fantasy.premierleague.com/api"
LIVE_SEASON = 2026
SEASON_GAMES = 38
POSITIONS = {1: "Goalkeeper", 2: "Defender", 3: "Midfield", 4: "Attack"}
# 2026/27 Champions League league phase (UEFA), as FPL team names. Stage 1 = league phase;
# raise a club's stage here as it progresses (3 last 16, 4 QF, 5 SF, 6 final).
UCL_STAGE = {"Arsenal": 1, "Aston Villa": 1, "Liverpool": 1, "Man City": 1, "Man Utd": 1}
CLOSE_NAME = 0.85  # minimum name similarity for the second matching pass


def fetch(path):
    request = urllib.request.Request(f"{FPL}/{path}", headers={"User-Agent": "transfer-market/0.1"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(request, context=context, timeout=60) as response:
        return json.load(response)


def latest_final_gameweek(events):
    final = [e["id"] for e in events if e["finished"] and e["data_checked"]]
    return max(final) if final else None


def fold(text):
    """Lowercase, accents stripped, letters only: 'Ødegaard' -> 'odegaard'."""
    text = unicodedata.normalize("NFD", str(text).lower().replace("ø", "o").replace("ß", "ss"))
    return re.sub(r"[^a-z ]", " ", "".join(c for c in text if not unicodedata.combining(c))).strip()


def close_birth_date(a, b):
    """Same date, within 2 days, or day and month swapped (a common data-entry slip)."""
    if pd.isna(a) or pd.isna(b):
        return False
    swapped = b.day <= 12 and a == b.replace(month=b.day, day=b.month)
    return abs((a - b).days) <= 2 or swapped


def match_players(elements, tm_players):
    """Add player_id + match_method to each FPL player."""
    tm = tm_players.assign(folded=tm_players.name.map(fold), dob=tm_players.date_of_birth.dt.normalize())
    by_dob = {d: g for d, g in tm.groupby("dob")}
    manual = pd.read_csv(MANUAL_MATCHES) if MANUAL_MATCHES.exists() else pd.DataFrame(columns=["fpl_code", "player_id"])
    manual = dict(zip(manual.fpl_code, manual.player_id))

    ids, methods = [], []
    for p in elements.itertuples():
        full = fold(f"{p.first_name} {p.second_name}")
        words = set(f"{full} {fold(p.web_name)}".split())
        dob = pd.to_datetime(p.birth_date) if p.birth_date else pd.NaT
        player_id, method = None, "unmatched"
        if p.code in manual:
            player_id, method = int(manual[p.code]), "manual"
        else:
            same_day = by_dob.get(dob, tm.iloc[0:0])
            scored = sorted(((len(words & set(n.split())), i) for n, i in zip(same_day.folded, same_day.player_id)), reverse=True)
            scored = [s for s in scored if s[0] > 0]
            if scored and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                player_id, method = int(scored[0][1]), "birth date + name"
            else:
                close = difflib.get_close_matches(full, tm.folded.tolist(), n=5, cutoff=CLOSE_NAME)
                cands = tm[tm.folded.isin(close) & tm.dob.map(lambda d: close_birth_date(d, dob) or pd.isna(dob))]
                if len(cands) == 1:
                    player_id, method = int(cands.player_id.iloc[0]), "close name + birth date"
        ids.append(player_id)
        methods.append(method)
    return elements.assign(player_id=pd.array(ids, dtype="Int64"), match_method=methods)


def league_table(fixtures, teams, gameweek):
    """Table after the given gameweek: points, then goal difference, then goals scored."""
    rows = []
    for f in fixtures:
        if not f["finished"] or f["event"] is None or f["event"] > gameweek:
            continue
        h, a = f["team_h_score"], f["team_a_score"]
        rows.append((f["team_h"], h, a))
        rows.append((f["team_a"], a, h))
    games = pd.DataFrame(rows, columns=["team", "scored", "conceded"])
    games["points"] = 3 * (games.scored > games.conceded) + (games.scored == games.conceded)
    table = games.groupby("team").agg(played=("points", "size"), points=("points", "sum"),
                                      scored=("scored", "sum"), conceded=("conceded", "sum"))
    table = table.reindex([t["id"] for t in teams], fill_value=0)
    table["gd"] = table.scored - table.conceded
    table = table.sort_values(["points", "gd", "scored"], ascending=False)
    table["position"] = range(1, len(table) + 1)
    return table


def season_totals(gameweek):
    """Each player's stats summed over gameweeks 1..gameweek from FPL's per-gameweek feed.

    The season totals in bootstrap-static update live during a gameweek, so they'd include
    matches past the last final gameweek; summing per gameweek keeps everything in step.
    """
    totals = {}
    for gw in range(1, gameweek + 1):
        for e in fetch(f"event/{gw}/live/")["elements"]:
            st = e["stats"]
            t = totals.setdefault(e["id"], dict.fromkeys(
                ["minutes", "goals", "assists", "yellows", "reds", "clean_sheets", "apps"], 0))
            t["minutes"] += st["minutes"]
            t["goals"] += st["goals_scored"]
            t["assists"] += st["assists"]
            t["yellows"] += st["yellow_cards"]
            t["reds"] += st["red_cards"]
            t["clean_sheets"] += st["clean_sheets"]
            # one entry per fixture played (a double gameweek has two)
            t["apps"] += sum(any(x["identifier"] == "minutes" and x["value"] > 0 for x in f["stats"]) for f in e["explain"])
    return pd.DataFrame.from_dict(totals, orient="index")


def build(bootstrap, fixtures, gameweek):
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    elements = pd.DataFrame(bootstrap["elements"]).drop(columns=["minutes", "goals_scored", "assists",
                                                                 "yellow_cards", "red_cards", "clean_sheets"])
    elements = elements.join(season_totals(gameweek), on="id")
    elements = elements[elements.minutes > 0].copy()  # only players who've played this season
    table = league_table(fixtures, bootstrap["teams"], gameweek)

    players, valuations, appearances, games, competitions = btt.load()
    elements = match_players(elements, players[["player_id", "name", "date_of_birth"]])
    MATCHES.parent.mkdir(parents=True, exist_ok=True)
    elements.assign(team=elements.team.map(teams))[
        ["code", "first_name", "second_name", "birth_date", "team", "minutes", "player_id", "match_method"]
    ].rename(columns={"code": "fpl_code"}).sort_values(["match_method", "team"]).to_csv(MATCHES, index=False)

    rows = elements[elements.player_id.notna()].copy()
    rows["player_id"] = rows.player_id.astype(int)
    rows["season"] = LIVE_SEASON
    rows["fpl_team"] = rows.team.map(teams)

    # FPL team -> Transfermarkt club: the club most of its matched players were at in June 2026
    tm_club = players.set_index("player_id")[["current_club_id", "current_club_name"]]
    rows = rows.join(tm_club, on="player_id")
    team_club = rows.groupby("fpl_team").current_club_id.agg(lambda c: c.mode().iloc[0])
    rows["club_id"] = rows.fpl_team.map(team_club).astype(int)
    club_names = pd.read_csv(btt.RAW / "clubs.csv.gz", usecols=["club_id", "name"]).set_index("club_id").name
    rows["club_name"] = rows.club_id.map(club_names).fillna(rows.fpl_team)

    pl_games = games[(games.competition_id == btt.LEAGUE) & (games.season == LIVE_SEASON - 1)]
    pl_last_season = set(pl_games.home_club_id) | set(pl_games.away_club_id)
    rows["club_in_pl_last_season"] = rows.club_id.isin(pl_last_season)
    rows["club_league_position"] = rows.team.map(table.position)
    rows["season_fraction"] = rows.team.map(table.played) / SEASON_GAMES
    rows["ucl_stage_reached"] = rows.fpl_team.map(UCL_STAGE).fillna(0).astype(int)

    tm_info = players.set_index("player_id")[["name", "date_of_birth", "position"]]
    rows = rows.drop(columns=["position"], errors="ignore").join(tm_info, on="player_id")
    rows["position"] = rows.position.where(rows.position.isin(POSITIONS.values()), rows.element_type.map(POSITIONS))
    rows["age"] = ((pd.Timestamp(f"{LIVE_SEASON}-08-01") - rows.date_of_birth).dt.days / 365.25).round(1)
    rows[["other_league_minutes", "other_league_goals", "other_league_assists"]] = 0

    # last season's stats and value history, by the same rules as the training table
    covered = set(competitions.loc[competitions.type == "domestic_league", "competition_id"])
    apps_table = appearances.merge(games[["game_id", "season"]], on="game_id")
    rows = btt.attach_previous_season(rows, apps_table, games, covered)
    rows = btt.attach_values(rows, valuations)
    rows["value_end"] = rows["value_end_age_days"] = None  # unknown until June 2027
    rows["usable_for_training"] = False
    rows["gameweek"] = gameweek

    columns = list(pd.read_csv(btt.OUT, nrows=0).columns) + [
        "season_fraction", "yellows", "reds", "clean_sheets", "gameweek", "fpl_team"]
    return rows[columns].sort_values("player_id").reset_index(drop=True), elements


def set_ci_output(changed):
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            out.write(f"changed={'true' if changed else 'false'}\n")


def main():
    bootstrap = fetch("bootstrap-static/")
    gameweek = latest_final_gameweek(bootstrap["events"])
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    new = gameweek is not None and gameweek != state.get("gameweek")
    if "--check" in sys.argv:
        print(f"latest final gameweek: {gameweek}; built: {state.get('gameweek')}; new: {new}")
        set_ci_output(new)
        return
    if not new and "--force" not in sys.argv:
        print(f"no new final gameweek (latest {gameweek}, already built)")
        set_ci_output(False)
        return
    if gameweek is None:
        print("no final gameweek yet this season")
        set_ci_output(False)
        return

    rows, elements = build(bootstrap, fetch("fixtures/"), gameweek)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT, index=False)
    unmatched = elements[elements.player_id.isna()]
    STATE.write_text(json.dumps({
        "season": LIVE_SEASON, "gameweek": gameweek,
        "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "players_with_minutes": int(len(elements)), "matched": int(elements.player_id.notna().sum()),
        "with_start_value": int(rows.value_1y_ago.notna().sum()),
        "unmatched": sorted(f"{p.first_name} {p.second_name}" for p in unmatched.itertuples()),
    }, indent=2) + "\n")
    print(f"gameweek {gameweek}: {len(rows)} matched players ({rows.value_1y_ago.notna().sum()} with a start value), "
          f"{len(unmatched)} unmatched -> {OUT.relative_to(ROOT)}")
    set_ci_output(True)


if __name__ == "__main__":
    main()
