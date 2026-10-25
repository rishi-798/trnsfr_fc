"""Rewrite each "why" explanation into two plain sentences with Gemini, ahead of time.

Models: gemini-3.8-flash, falling back to gemini-3.6-flash when it's overloaded or out of quota.

The site never calls Gemini: this script writes the text once and saves it, and the site
shows the saved text. Gemini only rephrases facts we give it (the template sentence, the
model's factor effects, key stats); every rewrite is checked and rejected if it:
  - uses a number that isn't in the facts,
  - mentions things the model doesn't know (injuries, contracts, fees, rumours, ...),
  - doesn't state the forecast's direction, or contradicts it in the first sentence,
  - isn't two sentences, or is too long.
Rejected players keep the template text.

Each saved rewrite carries a fingerprint of the facts (plus model and prompt version);
the exporter only shows it while the facts still match, and reruns only send players whose
facts changed. The season in progress goes first, then the most recent seasons, so a daily
quota running out leaves the most-viewed players done; rerun later to continue.

Needs GEMINI_API_KEY in the environment or in .env (never committed). Without it the
script prints a note and exits, and the site keeps the template text.

Usage:
  python scripts/write_explanations.py              write missing/changed rewrites
  python scripts/write_explanations.py --limit 40   only this many players (for testing)
  python scripts/write_explanations.py --dry-run    show one request's facts, call nothing
  python scripts/write_explanations.py --retry-rejected   also resend previously rejected ones

Reads site/data/players.json (run export_site_data.py first); writes
data/processed/explanations_ai.csv. Run export_site_data.py again afterwards.
"""
import hashlib
import json
import os
import re
import sys
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import certifi
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SITE_DATA = ROOT / "site" / "data" / "players.json"
CACHE = ROOT / "data" / "processed" / "explanations_ai.csv"
ENV_FILE = ROOT / ".env"

# tried in order when one is overloaded (503) or rate-limited. Flash-Lite is left out on purpose:
# in testing it copied the template's jargon instead of rephrasing it. If neither is available
# the run stops and the template stays until the next run.
MODELS = ["gemini-3.8-flash", "gemini-3.6-flash"]
MODEL = MODELS[0]  # shown in messages; each saved row records the model that actually wrote it
API = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
TIMEOUT = 240  # seconds per request
BUSY_ROUNDS, BUSY_WAIT = 3, 60  # when every model is busy: try this many rounds, this many seconds apart
PROMPT_VERSION = 4  # bump when the instructions change, so everything is rewritten
BATCH = 20  # players per request
MAX_WORDS = 70
FLAT = 0.03  # same as the site: smaller changes read as flat

GROUP_NAMES = {"fwd": "forward", "mid": "midfielder", "def": "defender", "gk": "goalkeeper"}
UCL = ["none", "league phase", "playoff", "last 16", "quarter-final", "semi-final", "final"]
# things the model knows nothing about; a rewrite mentioning them is making things up
OFF_LIMITS = re.compile(
    r"\b(injur\w*|fitness|contract\w*|rumou?r\w*|linked|bid\w*|fee\w*|agent\w*|transfer window|"
    r"world cup|euros\b|tournament\w*|international\w*|manager\w*|coach\w*|sacked|form guide)\b", re.I)
RISE_WORDS = re.compile(r"\b(rise|rises|rising|up|increase\w*|climb\w*|grow\w*|gain\w*|jump\w*|boost\w*|higher)\b", re.I)
FALL_WORDS = re.compile(r"\b(fall|falls|falling|drop\w*|down|decrease\w*|decline\w*|dip\w*|slide\w*|lower|lose\w*|slip\w*)\b", re.I)
FLAT_WORDS = re.compile(r"\b(steady|flat|unchanged|hold\w*|stable|little change|roughly the same|barely)\b", re.I)

INSTRUCTIONS = """You write the short "why" note for a football stats site that forecasts Premier League players'
Transfermarkt market values. For each player you get facts from a statistical model: the value at season start,
the model's predicted value at season end, the % change and its direction, the factors that pushed the forecast up
or down (each with its % effect on its own), a template sentence that already names the concrete reasons, and key stats.

Write exactly two sentences, together at most 60 words, in plain, friendly English for football fans:
1. First sentence: where the model expects the value to go, saying the direction in words (rise, climb, drop, fall,
   or stay roughly steady), the % change as a plain number, the predicted value, and the season-start value.
   Example: "The model expects Erling Haaland's value to drop 10% to €179m from €200m at the start of the season."
2. Second sentence: the one or two biggest reasons and what partly offsets them, saying which way each one pushes
   ("pulled down by", "held back by", "lifted by", "helped by"), in everyday words built from the template's details.
   Do not quote the factors' % effects and do not use + or - signs.

What the template phrases mean (rephrase them like this, never copy them word for word):
- "already rose €A → €B": the value already jumped from €A to €B, and the model expects big jumps to level off.
- "falling value €A → €B": the value has been sliding from €A to €B, and the model expects that to continue.
- "rising value €A → €B": the value has been climbing from €A to €B. "rebound after drop €A → €B": a bounce back is expected.
- "value steady at €X": the value has sat at €X. "room to grow from €X" / "already valued at €X": as it says.
- "N goals so far (worth less at this price)" or "N minutes so far (worth less at this price)": the goals or minutes
  count for less because the player is already so expensive (never imply the goals themselves are bad).
- "regular minutes (N so far)": "N minutes so far" or "playing regularly (N minutes so far)". "limited minutes (N)": only N minutes.
- "club Nth so far": the club (use its name) sitting Nth. "club finished Nth": the club finishing Nth.
- "attack/midfield/defender/goalkeeper position": playing in that position.
- "0.00 goals + assists per 90": no goals or assists yet. "X goals + assists per 90": X goals plus assists per 90 minutes.
- "age N": "being N" (e.g. "being 28"), never "being age N". "full history on record": an established value history (the model expects fewer jumps).
- "short value history" / "no last-season stats": little history to go on. "last season (...)": last season's numbers.
- "promoted club": playing for a newly promoted club. "established pl club": an established Premier League club.

Rules:
- Use only the facts given. Do not add anything else: no injuries, contracts, transfer fees, rumours, managers,
  international football or tournaments, and no opinions about the player's quality.
- Every number you write must appear in the facts, written the same way (e.g. "€179m", "10%", "529 minutes").
- Describe what the model weighed ("the model expects", "it weighs"), not real-world causes.
- If "season_in_progress" is true, the stats are this season so far: say "so far".
- Refer to the player by name; do not use pronouns for the player.
- No emojis, no exclamation marks.

Return JSON: a list with one object per player, {"id": <the player's id string>, "text": <the two sentences>}."""


def money(v):
    """Same format as the site: €179m, €1.5m, €850k."""
    if v >= 1e6:
        m = v / 1e6
        return f"€{round(m)}m" if m >= 10 else f"€{m:.1f}".rstrip("0").rstrip(".") + "m"
    return f"€{round(v / 1e3)}k"


def season_label(s):
    return f"{str(s)[2:]}/{str(s + 1)[2:]}"


def facts(p, live):
    """Everything Gemini may use for one player-season, already formatted as on the site."""
    start, predicted = p["values"][2], p["predicted"]
    change = predicted / start - 1
    in_progress = bool(live and p["season"] == live["season"])
    prev = p["prev"]
    return {
        "id": f"{p['id']}-{p['season']}",
        "name": p["name"],
        "season": season_label(p["season"]),
        "season_in_progress": in_progress,
        **({"after_gameweek": live["gameweek"]} if in_progress else {}),
        "club": p["club"],
        "position": GROUP_NAMES.get(p["group"], p["group"]),
        "age": int(p["age"]) if p["age"] is not None else None,
        "value_at_season_start": money(start),
        "predicted_value_at_season_end": money(predicted),
        "change": f"{'+' if change >= 0 else '-'}{abs(round(change * 100))}%",
        "direction": "up" if change > FLAT else "down" if change < -FLAT else "roughly unchanged",
        "factors": [{"factor": "club position so far" if in_progress and f["name"] == "club finish" else f["name"],
                     "effect_on_its_own": f"{'+' if f['effect'] >= 0 else '-'}{abs(round(f['effect'] * 100))}%"}
                    for f in p["factors"]],
        "template": p["explanation"],
        "stats": {
            "appearances": p["apps"], "minutes": p["minutes"], "goals": p["goals"], "assists": p["assists"],
            ("club_position_so_far" if in_progress else "club_final_position"): p["league_position"],
            "champions_league": UCL[p["ucl_stage"]], "promoted_club": p["promoted"],
            "last_season": None if prev["minutes"] is None else
            {"minutes": prev["minutes"], "goals": prev["goals"], "assists": prev["assists"]},
        },
        "value_history_each_june": [money(v) if v is not None else None for v in p["values"][:3]],
    }


def fingerprint(f):
    payload = json.dumps({"facts": f, "prompt": PROMPT_VERSION}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def numbers_in(text):
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def fact_values(obj, skip=("id",)):
    """All values in the facts as text (not field names, not the internal id), for the number check."""
    if isinstance(obj, dict):
        return " ".join(fact_values(v) for k, v in obj.items() if k not in skip)
    if isinstance(obj, list):
        return " ".join(fact_values(v) for v in obj)
    return "" if obj is None or isinstance(obj, bool) else str(obj)


def check(text, f):
    """Return None if the rewrite is acceptable, else the reason it was rejected."""
    text = (text or "").strip()
    if not text:
        return "empty"
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s]
    if len(sentences) != 2:
        return f"{len(sentences)} sentences"
    if len(text.split()) > MAX_WORDS:
        return "too long"
    extra = numbers_in(text) - numbers_in(fact_values(f))
    if extra:
        return f"numbers not in facts: {sorted(extra)}"
    if OFF_LIMITS.search(text):
        return f"off-limits topic: {OFF_LIMITS.search(text).group(0)}"
    first = sentences[0]
    want, avoid = {"up": (RISE_WORDS, FALL_WORDS), "down": (FALL_WORDS, RISE_WORDS)}.get(f["direction"], (FLAT_WORDS, None))
    if not want.search(first) and not (f["direction"] == "roughly unchanged" and (RISE_WORDS.search(first) or FALL_WORDS.search(first))):
        return "first sentence doesn't state the direction"
    if avoid and avoid.search(first) and not want.search(first):
        return "first sentence contradicts the direction"
    if f["predicted_value_at_season_end"] not in text:
        return "predicted value missing"
    return None


def load_cache():
    if CACHE.exists():
        return pd.read_csv(CACHE, dtype={"text": str, "reason": str}, keep_default_na=False)
    return pd.DataFrame(columns=["player_id", "season", "fingerprint", "status", "text", "reason", "model", "written_at"])


def attach(players, live):
    """For the exporter: add 'why_ai' to each player whose saved rewrite still matches its facts."""
    cache = load_cache()
    ok = cache[cache.status == "ok"].set_index(["player_id", "season"])
    shown = 0
    for p in players:
        p["why_ai"] = None
        if not p.get("explanation") or (p["id"], p["season"]) not in ok.index:
            continue
        row = ok.loc[(p["id"], p["season"])]
        if row.fingerprint == fingerprint(facts(p, live)):
            p["why_ai"] = row.text
            shown += 1
    return shown


def api_key():
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "GEMINI_API_KEY" and value.strip():
                return value.strip().strip('"').strip("'")
    return None


class Unavailable(Exception):
    """This model can't take the request right now (overloaded or out of quota)."""


SCHEMA = {"type": "array", "items": {"type": "object", "required": ["id", "text"],
                                     "properties": {"id": {"type": "string"}, "text": {"type": "string"}}}}


def ask_gemini(key, model, batch):
    """One request for a batch of players; returns {id: text}. Raises Unavailable on 429/503."""
    body = {
        "contents": [{"parts": [{"text": INSTRUCTIONS + "\n\nPlayers:\n" + json.dumps(batch, ensure_ascii=False, indent=1)}]}],
        "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA},
    }
    request = urllib.request.Request(API.format(model=model), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", "x-goog-api-key": key})
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urllib.request.urlopen(request, context=context, timeout=TIMEOUT) as response:
            out = json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")[:300]
        if error.code in (429, 503):
            raise Unavailable(f"{model}: HTTP {error.code} {detail}") from None
        raise RuntimeError(f"{model}: HTTP {error.code} {detail}") from None
    parts = out.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    items = json.loads(parts[0]["text"]) if parts else []
    return {str(item.get("id")): str(item.get("text", "")) for item in items if isinstance(item, dict)}


def main():
    args = sys.argv[1:]
    limit = int(args[args.index("--limit") + 1]) if "--limit" in args else None
    data = json.loads(SITE_DATA.read_text())
    live = data.get("live")
    cache = load_cache()
    done = {(r.player_id, r.season): r for r in cache.itertuples()}

    todo = []
    for p in data["players"]:
        if not p.get("explanation") or p.get("predicted") is None:
            continue
        f = facts(p, live)
        prior = done.get((p["id"], p["season"]))
        fp = fingerprint(f)
        if prior is not None and prior.fingerprint == fp and (prior.status == "ok" or "--retry-rejected" not in args):
            continue
        todo.append((p["season"], fp, f))
    # current season first, then newest seasons; within a season, most valuable first
    order = {f"{p['id']}-{p['season']}": p["values"][2] or 0 for p in data["players"]}
    todo.sort(key=lambda t: (-t[0], -order.get(t[2]["id"], 0)))
    if limit:
        todo = todo[:limit]
    print(f"{len(todo)} explanations to write ({' then '.join(MODELS)}, {BATCH} per request)")
    if not todo:
        return
    if "--dry-run" in args:
        print(json.dumps([t[2] for t in todo[:2]], ensure_ascii=False, indent=1))
        return

    key = api_key()
    if not key:
        print("no GEMINI_API_KEY in the environment or .env; keeping template explanations")
        return

    written = rejected = 0
    for start in range(0, len(todo), BATCH):
        batch = todo[start:start + BATCH]
        answers, used = None, None
        for round_ in range(BUSY_ROUNDS):  # overload spikes are usually short: wait, then retry
            if round_:
                print(f"  all models busy; waiting {BUSY_WAIT}s (try {round_ + 1} of {BUSY_ROUNDS})")
                time.sleep(BUSY_WAIT)
            for model in MODELS:  # fall back to the next model when one is overloaded or out of quota
                for attempt in range(2):
                    try:
                        answers, used = ask_gemini(key, model, [t[2] for t in batch]), model
                        break
                    except Unavailable as error:
                        print(f"  {str(error).splitlines()[0]} (busy or out of quota)")
                        break  # try the next model straight away
                    except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, KeyError) as error:
                        print(f"  {model} attempt {attempt + 1} failed: {error}")
                        time.sleep(5)
                if answers is not None:
                    break
            if answers is not None:
                break
        if answers is None:
            print(f"no model available after {written} written; progress saved, rerun later to continue")
            return
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows = []
        for season, fp, f in batch:
            text = (answers.get(f["id"]) or "").strip()
            reason = check(text, f)
            player_id = int(f["id"].rsplit("-", 1)[0])
            rows.append({"player_id": player_id, "season": season, "fingerprint": fp,
                         "status": "rejected" if reason else "ok", "text": text, "reason": reason or "",
                         "model": used, "written_at": now})
            written += reason is None
            rejected += reason is not None
        new = pd.DataFrame(rows)
        cache = pd.concat([cache[~cache.set_index(["player_id", "season"]).index.isin(
            new.set_index(["player_id", "season"]).index)], new], ignore_index=True)
        cache.sort_values(["season", "player_id"]).to_csv(CACHE, index=False)  # save after every batch
        print(f"  batch {start // BATCH + 1} ({used}): {len(batch) - sum(r['status'] == 'rejected' for r in rows)} ok, "
              f"{sum(r['status'] == 'rejected' for r in rows)} rejected")
    print(f"done: {written} written, {rejected} rejected (kept template) -> {CACHE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
