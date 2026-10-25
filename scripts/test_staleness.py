"""Compare staleness thresholds for training labels (90 / 180 / 365 days / no limit).

A training row is kept only if its end-of-season value is at most N days old at the June 30
cutoff. To compare thresholds fairly, only the training rule changes: every threshold is
scored on the same test rows, those whose end value is at most 90 days old (a valid answer
under every threshold). The current 180-day test set is reported as a cross-check.

Walk-forward like the site: each season is predicted by a model trained on earlier seasons.
Reads data/processed/player_seasons.csv; writes data/processed/staleness_test.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import train_model as tm

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "processed" / "staleness_test.csv"
THRESHOLDS = {"90 days": 90, "180 days (current)": 180, "365 days": 365, "no limit": np.inf}
TEST_SETS = {"labels <=90 days": 90, "labels <=180 days": 180}


def main():
    table = pd.read_csv(tm.TABLE)
    known = table[table.value_end.notna() & table.value_1y_ago.notna()]
    print("training rows available by threshold (all seasons):",
          {name: int((known.value_end_age_days <= days).sum()) for name, days in THRESHOLDS.items()})

    records = []
    for name, days in THRESHOLDS.items():
        for season in [2023, 2024, 2025]:
            train_rows = known[(known.season < season) & (known.value_end_age_days <= days)]
            model = tm.train(train_rows)
            for test_name, test_days in TEST_SETS.items():
                test = known[(known.season == season) & (known.value_end_age_days <= test_days)]
                predicted = test.value_1y_ago * np.exp(model.predict(tm.features(test)))
                records.append({
                    "threshold": name, "season": season, "test_set": test_name,
                    "train_rows": len(train_rows), "test_rows": len(test),
                    "male": np.mean(np.abs(np.log(predicted) - np.log(test.value_end))),
                    "mae_m": np.mean(np.abs(predicted - test.value_end)) / 1e6,
                })
    out = pd.DataFrame(records)
    out.round(4).to_csv(OUT, index=False)

    for test_name in TEST_SETS:
        part = out[out.test_set == test_name]
        pivot = part.pivot(index="threshold", columns="season", values="male")
        pivot["mean"] = pivot.mean(axis=1)
        pivot["train rows (25/26)"] = part[part.season == 2025].set_index("threshold").train_rows
        print(f"\nMALE on test rows with {test_name} (n per season: "
              f"{part.groupby('season').test_rows.first().to_dict()})")
        print(pivot.reindex(THRESHOLDS).round(4).to_string())


if __name__ == "__main__":
    main()
