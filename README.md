# transfer fc

Explore Premier League player performance and estimated market values across the 2022/23–2025/26 seasons. A Python pipeline builds player-season records from Transfermarkt data, trains a valuation model, and exports results to a static HTML, CSS, and JavaScript browser.

## Features

- Browse players by season and position, sorted by value or performance score.
- Inspect season statistics, valuation history, and model estimates in player detail panels.
- See per-player explanations derived from SHAP contributions.
- Compare the model with a baseline that keeps each player's starting value unchanged.
- Review data quality with a stratified sample of up to 50 player-seasons.

## Run the site

If `site/data/players.json` is present, you can browse the exported results without installing the modeling dependencies. From the repository root:

```sh
python3 -m http.server 8000 --bind 127.0.0.1 --directory site
```

Open [http://localhost:8000](http://localhost:8000). Serve the site over HTTP because it loads its data with `fetch()`. No Node.js build step or backend service is required. Press `Ctrl+C` to stop the server.

## Set up the Python pipeline

Use Python 3 with `venv` and `pip`. The dependencies are listed in [requirements.txt](requirements.txt): pandas, NumPy, scikit-learn, SHAP, certifi, and Jupyter. Dependency versions are currently unpinned.

```sh
git clone https://github.com/rishi-798/trnsfr_fc.git
cd trnsfr_fc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate` instead. If you already have a checkout, start with the environment creation step in its root directory.

## Build the data and predictions

Run these commands in order from the repository root:

```sh
python scripts/download_data.py
python scripts/build_training_table.py
python scripts/train_model.py
python scripts/export_site_data.py
```

| Step | Output |
| --- | --- |
| Download | Six compressed CSV tables in `data/raw/`: players, valuations, appearances, games, clubs, and competitions |
| Build training table | `data/processed/player_seasons.csv` |
| Train and evaluate | `data/processed/predictions.csv`, `explanations.csv`, and `model_metrics.json` |
| Export site data | `site/data/players.json` |

The downloader needs internet access and skips files that already exist. To refresh a table or recover an interrupted download, move or remove the corresponding raw file before rerunning it. Raw data is excluded from Git.

The build, training, and export steps replace their generated outputs. Rerun the downstream steps after changing the raw data or feature logic to keep the site consistent. The exporter also works before training, with observed values and performance scores but no model predictions.

For exploratory analysis:

```sh
jupyter notebook notebooks/01_explore.ipynb
```

## Data conventions

- Each row represents a player with at least one Premier League appearance in a season. Seasons use their starting year: `2025` means 2025/26.
- `value_end` is the latest available valuation on or before June 30 after the season ends. Historical values use the same cutoff one, two, and three years earlier. All values are in euros.
- A row is marked `usable_for_training` only when its ending valuation exists and is no more than 180 days old at the cutoff. Training additionally requires a starting valuation.
- Current-season league statistics cover Premier League appearances. Appearances in other covered domestic leagues are stored separately as `other_league_*` fields.
- The player's main club is the club for which they played the most Premier League minutes that season.
- Previous-season totals combine all covered domestic leagues. `prev_league` identifies the league with the most minutes; it does not restrict those totals to that league.
- Missing history remains missing, with reason columns. Previous-season league statistics become zero only when other recorded appearances establish participation for a club in a covered league without recorded league minutes.

## Model and evaluation

The model is a 500-tree `RandomForestRegressor` with random seed 42. It uses age, position, playing time, goals, assists, club context, previous-season performance, and valuation history to estimate the logarithmic change in value:

```text
target = log(value_end / value_1y_ago)
predicted_value = value_1y_ago × exp(predicted_log_change)
```

Because the features include the completed season's performance, these are retrospective season-end estimates, not preseason forecasts or predictions of transfer fees.

Evaluation trains on 2022/23–2024/25 and tests on 2025/26. The saved [metrics](data/processed/model_metrics.json) report 1,560 training rows and 506 test rows:

| Metric on 2025/26 | Model | Unchanged-value baseline |
| --- | ---: | ---: |
| Mean absolute log error (MALE) | 0.2640 | 0.4808 |
| Mean absolute error (EUR) | €5,245,254 | €8,179,595 |

Lower is better for both metrics. These numbers describe the saved run; regenerating data or changing dependencies can change the results. The metrics file also contains subgroup scores and permutation feature importance.

Site predictions use walk-forward training: each season is predicted using only earlier seasons. There are no model predictions for 2022/23 because the training table contains no earlier season, or for players without a starting valuation. The site falls back to the observed ending value when a prediction is unavailable. Grouped SHAP contributions explain each estimate in log-change units; they describe the model's behavior, not causal effects.

## Performance score

The browsing score is a separate, hand-built measure. It uses position-specific weights on per-90 statistics:

| Position | Positive contributions |
| --- | --- |
| Forwards | `5 × goals + 4 × assists` |
| Midfielders | `3 × goals + 4 × assists` |
| Defenders | `3 × clean sheets + 3 × goals + 3 × assists` |
| Goalkeepers | `5 × clean sheets` |

Every position subtracts `2 × yellow cards + 5 × red cards`, also per 90 minutes. The result is multiplied by `minutes / (minutes + 900)` to reduce the influence of small samples. A clean sheet requires at least 60 minutes played and zero team goals conceded. Rankings and percentiles compare players within the same season and position group.

## Data review and limitations

After building the training table, generate a review sample with:

```sh
python scripts/make_pilot_sample.py
```

This writes `data/processed/pilot_sample.csv` with `review_status` and `review_notes` columns. It refuses to overwrite an existing sample with review entries. Preserve that file before regenerating; `--force` explicitly discards its review entries.

The [existing pilot review](data/processed/pilot_review_2026-10-17.md) records 20 reviewed rows, including three valuation discrepancies. The review did not change the underlying labels. Its deliberately selected sample is not an estimate of the full dataset's error rate.

League coverage and valuation coverage are incomplete, so missing history does not necessarily mean a player did not play. The exploration notebook records a valuation cutoff of June 12, 2026 for the local source snapshot. These gaps should be considered when interpreting model estimates and scores.

## Project layout

```text
data/
  raw/                         Downloaded source tables (ignored by Git)
  processed/                   Training data, predictions, metrics, and reviews
notebooks/
  01_explore.ipynb              Initial exploration and coverage checks
scripts/
  download_data.py              Download source tables
  build_training_table.py       Build player-season features and labels
  make_pilot_sample.py          Generate a manual review sample
  train_model.py                Train, evaluate, and explain predictions
  export_site_data.py           Export browser data and performance scores
site/
  index.html                   Static site entry point
  style.css                    Site styling
  app.js                       Filtering, sorting, and player detail views
  data/players.json            Generated site dataset
requirements.txt               Python dependencies
```

## Data source

The downloader uses the public tables from [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), derived from Transfermarkt. See the upstream project for dataset documentation and licensing terms.
