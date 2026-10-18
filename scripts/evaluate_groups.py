"""Break down model vs baseline errors by season and player group.

Uses the walk-forward predictions (each season predicted by a model trained only on earlier
seasons), scored on rows usable for training. Baseline = value unchanged (value_1y_ago).
Value bands use the starting value, which is known before the season; banding by the
actual ending value would put surprise risers in the expensive band because of the rise.

Reads data/processed/player_seasons.csv and predictions.csv; writes
data/processed/error_breakdown.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "processed" / "player_seasons.csv"
PREDICTIONS = ROOT / "data" / "processed" / "predictions.csv"
BREAKDOWN = ROOT / "data" / "processed" / "error_breakdown.csv"

VALUE_BANDS = [0, 5e6, 15e6, 40e6, np.inf]
VALUE_LABELS = ["start_<5m", "start_5-15m", "start_15-40m", "start_40m+"]


def scored_rows():
    table = pd.read_csv(TABLE)
    preds = pd.read_csv(PREDICTIONS)
    rows = table.merge(preds, on=["player_id", "season"], how="inner")
    rows = rows[rows.usable_for_training & rows.value_1y_ago.notna()].copy()
    rows["value_band"] = pd.cut(rows.value_1y_ago, VALUE_BANDS, labels=VALUE_LABELS, right=False)
    return rows


def groups(rows):
    yield "all", rows.index == rows.index
    for p in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
        yield f"position_{p.lower()}", rows.position == p
    for band in VALUE_LABELS:
        yield band, rows.value_band == band
    yield "promoted_club", ~rows.club_in_pl_last_season.astype(bool)
    yield "missing_prev_stats", rows.prev_minutes.isna()


def score(rows):
    actual = rows.value_end
    model, base = rows.predicted_value, rows.value_1y_ago
    return {
        "n": len(rows),
        "model_male": np.mean(np.abs(np.log(model) - np.log(actual))),
        "base_male": np.mean(np.abs(np.log(base) - np.log(actual))),
        "model_mae_m": np.mean(np.abs(model - actual)) / 1e6,
        "base_mae_m": np.mean(np.abs(base - actual)) / 1e6,
    }


def main():
    rows = scored_rows()
    records = []
    for season, season_rows in [("pooled", rows), *rows.groupby("season")]:
        for name, mask in groups(season_rows):
            subset = season_rows[mask]
            if len(subset):
                records.append({"season": season, "group": name, **score(subset)})
    out = pd.DataFrame(records)
    out["male_gain_pct"] = 100 * (1 - out.model_male / out.base_male)
    out.round(4).to_csv(BREAKDOWN, index=False)

    pd.set_option("display.width", 160)
    for season, part in out.groupby("season", sort=False):
        print(f"\n== {season} ==")
        print(part.drop(columns="season").set_index("group").round(3).to_string())


if __name__ == "__main__":
    main()
