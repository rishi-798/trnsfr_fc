"""Train and evaluate the market value model.

Target: log(value_end / value_1y_ago), the change from the player's value at the start of
the season. Predicted value = value_1y_ago * exp(prediction).

Evaluation: train on 2022/23-2024/25, test once on 2025/26. Primary metric MALE (mean
absolute log error), secondary MAE in euros, both against the baseline "value unchanged".

Site predictions are walk-forward: each season is predicted by a model trained only on
earlier seasons (2022/23 has none), so no shown prediction has seen its own answer.

Reads data/processed/player_seasons.csv; writes data/processed/predictions.csv and
data/processed/model_metrics.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "processed" / "player_seasons.csv"
PREDICTIONS = ROOT / "data" / "processed" / "predictions.csv"
METRICS = ROOT / "data" / "processed" / "model_metrics.json"
EXPLANATIONS = ROOT / "data" / "processed" / "explanations.csv"

# SHAP contributions are summed into these human-level reasons for the explanations
REASON_GROUPS = {
    "age": ["age"], "minutes": ["minutes"], "goals": ["goals"], "assists": ["assists"],
    "start_value": ["value_1y_ago"],
    "value_history": ["value_2y_ago", "value_3y_ago"],
    "valuation_age": ["value_1y_ago_age_days", "value_2y_ago_age_days", "value_3y_ago_age_days"],
    "club_finish": ["club_league_position"], "ucl": ["ucl_stage_reached"], "promoted": ["promoted_club"],
    "last_season": ["prev_minutes", "prev_goals", "prev_assists"],
    "other_league": ["other_league_minutes", "other_league_goals", "other_league_assists"],
}
REASON_PREFIXES = {"position": "position_", "prev_league": "prev_league_",
                   "missing_history": ("prev_stats_missing_", "value_2y_ago_missing", "value_3y_ago_missing")}

TEST_SEASON = 2025
SEED = 42
N_TREES = 500

NUMERIC = [
    "age", "minutes", "goals", "assists",
    "other_league_minutes", "other_league_goals", "other_league_assists",
    "club_league_position", "ucl_stage_reached",
    "prev_minutes", "prev_goals", "prev_assists",
    "value_1y_ago", "value_2y_ago", "value_3y_ago",
    "value_1y_ago_age_days", "value_2y_ago_age_days", "value_3y_ago_age_days",
]
PREV_LEAGUES = ["GB1", "ES1", "L1", "IT1", "FR1"]  # anything else -> prev_league_other
# Kept out on purpose: identifiers (player_id, name, season, club_*), the answer (value_end),
# and anything derived from it (value_end_age_days, usable_for_training).


def features(df):
    X = df[NUMERIC].astype(float).copy()
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
    return X


def target(df):
    return np.log(df.value_end / df.value_1y_ago)


def train(df):
    model = RandomForestRegressor(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
    model.fit(features(df), target(df))
    return model


def predict_values(model, df):
    return df.value_1y_ago * np.exp(model.predict(features(df)))


def explain(model, df):
    """Per-row SHAP contributions (log-change units), grouped into reasons.

    For every row: base + sum(reasons) == the model's predicted log change.
    """
    X = features(df)
    explainer = shap.TreeExplainer(model)
    values = pd.DataFrame(explainer.shap_values(X, check_additivity=False), columns=X.columns, index=df.index)
    gap = np.abs(explainer.expected_value + values.sum(axis=1) - model.predict(X)).max()
    assert gap < 1e-6, f"SHAP contributions don't add up to the prediction (gap {gap})"

    groups = dict(REASON_GROUPS)
    for reason, prefix in REASON_PREFIXES.items():
        groups[reason] = [c for c in X.columns if c.startswith(prefix)]
    assigned = [c for cols in groups.values() for c in cols]
    assert sorted(assigned) == sorted(X.columns), "every feature must belong to exactly one reason"

    reasons = pd.DataFrame({reason: values[cols].sum(axis=1) for reason, cols in groups.items()})
    reasons.insert(0, "base", float(np.ravel(explainer.expected_value)[0]))
    return reasons.round(4)


def scores(actual, predicted):
    return {
        "male": round(float(np.mean(np.abs(np.log(predicted) - np.log(actual)))), 4),
        "mae_eur": round(float(np.mean(np.abs(predicted - actual)))),
        "n": int(len(actual)),
    }


def main():
    table = pd.read_csv(TABLE)
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
