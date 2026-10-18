"""Train and evaluate the market value model.

Model: linear regression (ridge, i.e. with a small penalty on large weights) on
standardized features. Values enter on a log scale, value trends as log ratios, and age
with a squared term so it can rise then fall.

Target: log(value_end / value_1y_ago), the change from the player's value at the start of
the season. Predicted value = value_1y_ago * exp(prediction).

Evaluation: train on 2022/23-2024/25, test once on 2025/26. Primary metric MALE (mean
absolute log error), secondary MAE in euros, both against the baseline "value unchanged".

Site predictions are walk-forward: each season is predicted by a model trained only on
earlier seasons (2022/23 has none), so no shown prediction has seen its own answer.

The live season in progress (data/processed/live_season.csv, from update_live.py) is
predicted too. The model learned from full-season totals, so this season's minutes, goals
and assists are projected to a full season (x 1 / season_fraction) and blended with last
season's: 60% this season's pace, 40% last season (pace alone if last season is missing).

Reads data/processed/player_seasons.csv; writes data/processed/predictions.csv and
data/processed/model_metrics.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "processed" / "player_seasons.csv"
LIVE = ROOT / "data" / "processed" / "live_season.csv"
PREDICTIONS = ROOT / "data" / "processed" / "predictions.csv"
METRICS = ROOT / "data" / "processed" / "model_metrics.json"
EXPLANATIONS = ROOT / "data" / "processed" / "explanations.csv"

# Per-feature contributions are summed into these human-level reasons for the explanations
REASON_GROUPS = {
    "age": ["age", "age_sq"], "minutes": ["minutes", "minutes_x_value"], "goals": ["goals", "goals_x_value"],
    "assists": ["assists"],
    "scoring_rate": ["ga_per90"],
    # start value and its trends are strongly linked, so they're explained together
    # (they stay separate model features)
    "market_value_history": ["log_value", "trend_1y", "trend_2y"],
    "valuation_age": ["value_1y_ago_age_days", "value_2y_ago_age_days", "value_3y_ago_age_days"],
    "club_finish": ["club_league_position"], "ucl": ["ucl_stage_reached"], "promoted": ["promoted_club"],
    "last_season": ["prev_minutes", "prev_goals", "prev_assists"],
    "other_league": ["other_league_minutes", "other_league_goals", "other_league_assists"],
}
REASON_PREFIXES = {"position": "position_", "prev_league": "prev_league_",
                   "missing_history": ("prev_stats_missing_", "value_2y_ago_missing", "value_3y_ago_missing")}

TEST_SEASON = 2025
SEED = 42
ALPHAS = np.logspace(-2, 3, 30)  # ridge penalty strengths tried; the best is picked by cross-validation
PEAK_AGE = 26  # centre of the age curve
THIS_SEASON_WEIGHT = 0.6  # season in progress: weight on this season's pace vs last season
# The more a player is already worth, the less extra minutes and goals add (values flatten
# out near the top: €100m+ players rarely rise). Applied between €20m and €100m and flat
# above, so the few €100m+ players in the data can't push it too far.
DAMPING_FROM, DAMPING_TO = 20e6, 100e6

NUMERIC = [
    "age", "minutes", "goals", "assists",
    "other_league_minutes", "other_league_goals", "other_league_assists",
    "club_league_position", "ucl_stage_reached",
    "prev_minutes", "prev_goals", "prev_assists",
    "value_1y_ago_age_days", "value_2y_ago_age_days", "value_3y_ago_age_days",
]
PREV_LEAGUES = ["GB1", "ES1", "L1", "IT1", "FR1"]  # anything else -> prev_league_other
# Kept out on purpose: identifiers (player_id, name, season, club_*), the answer (value_end),
# and anything derived from it (value_end_age_days, usable_for_training).


def features(df):
    X = df[NUMERIC].astype(float).copy()
    # season in progress (season_fraction < 1): project this season's totals to a full
    # season, then blend with last season so a hot or cold start doesn't dominate
    if "season_fraction" in df:
        fraction = df["season_fraction"].fillna(1.0)
        in_progress = fraction < 1
        for col in ["minutes", "goals", "assists"]:
            pace = X[col] / fraction
            blended = THIS_SEASON_WEIGHT * pace + (1 - THIS_SEASON_WEIGHT) * df[f"prev_{col}"]
            X[col] = np.where(in_progress, blended.fillna(pace), X[col])
    # linear models need relationships that are roughly straight lines, so reshape:
    X["log_value"] = np.log(df.value_1y_ago)  # a €10m change matters more at €5m than at €100m
    X["trend_1y"] = np.log(df.value_1y_ago / df.value_2y_ago)  # last year's move
    X["trend_2y"] = np.log(df.value_2y_ago / df.value_3y_ago)  # the year before
    X["age_sq"] = (df.age - PEAK_AGE) ** 2  # lets value rise with age then fall
    X["ga_per90"] = (df.goals + df.assists) / np.maximum(df.minutes, 90) * 90
    X["promoted_club"] = (~df.club_in_pl_last_season.astype(bool)).astype(int)
    for pos in ["Attack", "Midfield", "Defender", "Goalkeeper"]:
        X[f"position_{pos.lower()}"] = (df.position == pos).astype(int)
    for league in PREV_LEAGUES:
        X[f"prev_league_{league}"] = (df.prev_league == league).astype(int)
    X["prev_league_other"] = (df.prev_league.notna() & ~df.prev_league.isin(PREV_LEAGUES)).astype(int)
    for reason in ["no_stats_in_dataset", "no_valuation"]:
        X[f"prev_stats_missing_{reason}"] = (df.prev_stats_missing_reason == reason).astype(int)
    X["value_2y_ago_missing"] = df.value_2y_ago.isna().astype(int)
    X["value_3y_ago_missing"] = df.value_3y_ago.isna().astype(int)
    # how far above €20m the player starts (log scale, capped at €100m; 0 below €20m)
    above = np.clip(X["log_value"] - np.log(DAMPING_FROM), 0, np.log(DAMPING_TO / DAMPING_FROM))
    X["minutes_x_value"] = X["minutes"] * above
    X["goals_x_value"] = X["goals"] * above
    # linear regression can't take blanks: missing history -> 0, flagged by the indicators above
    return X.fillna(0)


def target(df):
    return np.log(df.value_end / df.value_1y_ago)


class ClipToTrainingRange(BaseEstimator, TransformerMixin):
    """Cap each feature at the range seen in training. A straight line keeps going, so
    without this an early-season projection (e.g. 8 goals in 6 games -> 51) beyond any real
    season would be extrapolated into an extreme prediction."""

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_, self.high_ = X.min(axis=0), X.max(axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def train(df):
    model = make_pipeline(ClipToTrainingRange(), StandardScaler(), RidgeCV(alphas=ALPHAS))
    model.fit(features(df), target(df))
    return model


def predict_values(model, df):
    return df.value_1y_ago * np.exp(model.predict(features(df)))


def explain(model, df):
    """Per-row contributions (log-change units), grouped into reasons.

    For a linear model these are exact: weight x how far the player is from the training
    average (in standardized units). For every row: base + sum(reasons) == predicted log change.
    """
    X = features(df)
    ridge = model[-1]
    values = pd.DataFrame(model[:-1].transform(X) * ridge.coef_, columns=X.columns, index=df.index)
    gap = np.abs(ridge.intercept_ + values.sum(axis=1) - model.predict(X)).max()
    assert gap < 1e-6, f"contributions don't add up to the prediction (gap {gap})"

    groups = dict(REASON_GROUPS)
    for reason, prefix in REASON_PREFIXES.items():
        groups[reason] = [c for c in X.columns if c.startswith(prefix)]
    assigned = [c for cols in groups.values() for c in cols]
    assert sorted(assigned) == sorted(X.columns), "every feature must belong to exactly one reason"

    reasons = pd.DataFrame({reason: values[cols].sum(axis=1) for reason, cols in groups.items()})
    reasons.insert(0, "base", float(ridge.intercept_))
    return reasons.round(4)


def scores(actual, predicted):
    return {
        "male": round(float(np.mean(np.abs(np.log(predicted) - np.log(actual)))), 4),
        "mae_eur": round(float(np.mean(np.abs(predicted - actual)))),
        "n": int(len(actual)),
    }


def main():
    table = pd.read_csv(TABLE)
    if LIVE.exists():  # predicted like any later season; it has no answer yet, so never trained on
        table = pd.concat([table, pd.read_csv(LIVE)], ignore_index=True)
    has_start = table.value_1y_ago.notna()
    labelled = table[table.usable_for_training & has_start]

    train_rows = labelled[labelled.season < TEST_SEASON]
    test_rows = labelled[labelled.season == TEST_SEASON].copy()
    model = train(train_rows)

    test_rows["model"] = predict_values(model, test_rows)
    test_rows["baseline"] = test_rows.value_1y_ago

    groups = {
        "all": test_rows.index == test_rows.index,
        "promoted_club": ~test_rows.club_in_pl_last_season,
        "missing_prev_stats": test_rows.prev_minutes.isna(),
        **{f"position_{p.lower()}": test_rows.position == p for p in ["Attack", "Midfield", "Defender", "Goalkeeper"]},
    }
    results = {}
    for name, mask in groups.items():
        rows = test_rows[mask]
        results[name] = {
            "model": scores(rows.value_end, rows.model),
            "baseline": scores(rows.value_end, rows.baseline),
        }

    # Which features matter: how much worse the log error gets when each one is shuffled
    imp = permutation_importance(
        model, features(test_rows), target(test_rows),
        scoring="neg_mean_absolute_error", n_repeats=5, random_state=SEED, n_jobs=-1,
    )
    importances = (
        pd.Series(imp.importances_mean, index=features(test_rows).columns)
        .sort_values(ascending=False).round(4)
    )

    # Walk-forward predictions for the site
    predictions, explanations = [], []
    for season in sorted(table.season.unique()):
        history = labelled[labelled.season < season]
        if history.empty:
            continue
        season_model = model if season == TEST_SEASON else train(history)
        rows = table[(table.season == season) & has_start]
        predictions.append(pd.DataFrame({
            "player_id": rows.player_id, "season": season,
            "predicted_value": predict_values(season_model, rows).round(-4),
            "trained_on": f"{history.season.min()}-{history.season.max()}",
        }))
        explanations.append(explain(season_model, rows).assign(player_id=rows.player_id, season=season))
    pd.concat(predictions).to_csv(PREDICTIONS, index=False)
    pd.concat(explanations).to_csv(EXPLANATIONS, index=False)

    metrics = {
        "test_season": TEST_SEASON,
        "train_rows": int(len(train_rows)), "test_rows": int(len(test_rows)),
        "excluded_no_start_value": int((table.usable_for_training & ~has_start).sum()),
        "results": results,
        "importances": importances.to_dict(),
    }
    METRICS.write_text(json.dumps(metrics, indent=2))

    print(f"train rows {len(train_rows)}, test rows {len(test_rows)}, "
          f"excluded (no start value) {metrics['excluded_no_start_value']}")
    print(f"\n{'group':<22}{'n':>5}{'model MALE':>12}{'base MALE':>11}{'model MAE':>13}{'base MAE':>12}")
    for name, r in results.items():
        m, b = r["model"], r["baseline"]
        print(f"{name:<22}{m['n']:>5}{m['male']:>12.3f}{b['male']:>11.3f}"
              f"{m['mae_eur'] / 1e6:>12.2f}m{b['mae_eur'] / 1e6:>11.2f}m")
    print("\ntop features:\n" + importances.head(10).to_string())


if __name__ == "__main__":
    main()
