# transfer fc

Explore Premier League player performance and estimated market values from 2022/23 to the current 2026/27 season. A Python pipeline builds player-season records from Transfermarkt data, adds the season in progress from the Fantasy Premier League feed after every gameweek, trains a valuation model, and exports results to a static HTML, CSS, and JavaScript browser.

## Features

- Browse players by season and position, sorted by value or performance score, or search by player or club.
- Follow the current 2026/27 season with forecasts refreshed after every gameweek.
- Inspect season statistics, valuation history, and model estimates in player detail panels.
- See per-player explanations built from each factor's exact contribution to the estimate.
- Compare the model with a baseline that keeps each player's starting value unchanged.
- Review data quality with a stratified sample of up to 50 player-seasons.

## Run the site

If `site/data/players.json` is present, you can browse the exported results without installing the modeling dependencies. From the repository root:

```sh
python3 scripts/serve_site.py
```

Open [http://localhost:8100](http://localhost:8100). Serve the site over HTTP because it loads its data with `fetch()`. The script tells the browser to recheck files on every load, so regenerated data and code changes appear without a hard refresh; pass a different port as an argument if 8100 is taken. No Node.js build step or backend service is required. Press `Ctrl+C` to stop the server.

## Set up the Python pipeline

Use Python 3 with `venv` and `pip`. The dependencies are listed in [requirements.txt](requirements.txt): pandas, NumPy, scikit-learn, certifi, and Jupyter. Dependency versions are currently unpinned.

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
python scripts/update_live.py
python scripts/train_model.py
python scripts/export_site_data.py
```

`update_live.py` adds the current season from the Fantasy Premier League feed (see [Live 2026/27 season](#live-202627-season)); skip it to build the historical seasons only.

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
- The site omits player-seasons without a season-start valuation, since they have no price or prediction to show; performance-score ranks count only the players shown.
- Missing history remains missing, with reason columns. Previous-season league statistics become zero only when other recorded appearances establish participation for a club in a covered league without recorded league minutes.

## Model and evaluation

The model is a linear regression with a small ridge penalty (`StandardScaler` + `RidgeCV`, penalty chosen by cross-validation). It uses age, position, playing time, goals, assists, club context, previous-season performance, and valuation history to estimate the logarithmic change in value. Because a linear model can only fit straight-line relationships, some features are reshaped first: the starting value enters as a logarithm, value trends as log ratios between consecutive June 30 values, age with a squared term (centred on 26) so value can rise and then fall, and goals plus assists per 90 minutes. Minutes and goals are also multiplied by how far the starting value is above €20m (on a log scale, capped at €100m), so the model can learn that the same performance adds less to an already expensive player; without this it overrated the most expensive players, whose values rarely rise. Missing history is filled with 0 and flagged by indicator columns.

```text
target = log(value_end / value_1y_ago)
predicted_value = value_1y_ago × exp(predicted_log_change)
```

Because the features include the completed season's performance, these are retrospective season-end estimates, not preseason forecasts or predictions of transfer fees.

Evaluation trains on 2022/23–2024/25 and tests on 2025/26. The saved [metrics](data/processed/model_metrics.json) report 1,560 training rows and 506 test rows:

| Metric on 2025/26 | Model | Unchanged-value baseline |
| --- | ---: | ---: |
| Mean absolute log error (MALE) | 0.2723 | 0.4808 |
| Mean absolute error (EUR) | €5,201,563 | €8,179,595 |

Lower is better for both metrics. These numbers describe the saved run; regenerating data or changing dependencies can change the results. The metrics file also contains subgroup scores and permutation feature importance. A 500-tree random forest scored 0.2640 MALE on the same split; the linear model was chosen for its transparency. Run `python scripts/evaluate_groups.py` for errors by season, position, starting-value range, promoted clubs, and missing previous-season stats.

Site predictions use walk-forward training: each season is predicted using only earlier seasons. There are no model predictions for 2022/23 because the training table contains no earlier season, or for players without a starting valuation. The site's "actual" value is the market value at season start, and tiles fall back to it when a prediction is unavailable. Each estimate is explained by grouped per-feature contributions in log-change units: for a linear model these are exact (weight × the player's standardized distance from the training average) and add up to the prediction. They describe the model's behavior, not causal effects.

## Live 2026/27 season

The Transfermarkt dataset's updates are paused, and it ends with 2025/26. Everything the model needs from before the current season (the 30 June 2026 starting value, value history, and 2025/26 statistics) is already in it. [`scripts/update_live.py`](scripts/update_live.py) adds the season in progress from the public [Fantasy Premier League](https://fantasy.premierleague.com) feed: squads, positions, minutes, goals, assists, cards, clean sheets, and the league table.

- It rebuilds only when FPL marks a new gameweek as final (finished and data checked), summing each player's per-gameweek stats up to that gameweek so totals and games played stay in step. `--check` reports whether there is a new gameweek, and `--force` rebuilds anyway.
- FPL players are matched to Transfermarkt by birth date and shared name words, then by close name with birth dates within two days (or day and month swapped). Add stubborn cases to [`data/live/fpl_matches.csv`](data/live/fpl_matches.csv) (`fpl_code`, `player_id`); review every match in `data/live/matches.csv`. Players who aren't in the Transfermarkt dataset, mostly Championship-only players and teenagers, have no starting value and are left out.
- The model learned from full seasons, so this season's minutes, goals, and assists are projected to a full season (× 38 ÷ the club's games played) and blended with last season's totals: 60% this season's pace, 40% last season, so a hot or cold start doesn't dominate. Players without last-season statistics use this season's pace alone. Every feature is also capped at the range seen in training, so an early-season projection can't be extrapolated beyond any real season. Early forecasts are still noisy and settle as the season goes on.
- Champions League participation isn't in the FPL feed. The five English clubs in the 2026/27 league phase are listed in `UCL_STAGE` in `update_live.py`; raise a club's stage there as it progresses.
- There are no in-season valuations, and the season's actual end value won't exist until June 2027, so 2026/27 forecasts can't be scored yet.

A [GitHub Actions workflow](.github/workflows/refresh-live.yml) runs the check daily at 07:00 UTC. When there is a new final gameweek it rebuilds, retrains, re-exports, and commits the refreshed data to `main`; pull before working locally. Run it by hand from the repository's Actions tab, with an option to force a rebuild.

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
  processed/                   Training data, live season rows, predictions, metrics, and reviews
  live/                        Live-season state, FPL match list, and manual matches
notebooks/
  01_explore.ipynb              Initial exploration and coverage checks
scripts/
  download_data.py              Download source tables
  build_training_table.py       Build player-season features and labels
  make_pilot_sample.py          Generate a manual review sample
  train_model.py                Train, evaluate, and explain predictions
  update_live.py                Add the current season from the Fantasy Premier League feed
  export_site_data.py           Export browser data and performance scores
  evaluate_groups.py            Compare model and baseline errors by season and group
  serve_site.py                 Serve the site locally without stale browser caching
site/
  index.html                   Static site entry point
  style.css                    Site styling
  app.js                       Filtering, sorting, and player detail views
  data/players.json            Generated site dataset
.github/workflows/
  refresh-live.yml             Daily check that rebuilds after each final gameweek
requirements.txt               Python dependencies
```

## Data source

The downloader uses the public tables from [dcaribou/transfermarkt-datasets](https://github.com/dcaribou/transfermarkt-datasets), derived from Transfermarkt. See the upstream project for dataset documentation and licensing terms.

Live 2026/27 statistics come from the public Fantasy Premier League API (`fantasy.premierleague.com/api`). It is unofficial and undocumented, and its terms of use for republishing data have not been confirmed; this repository commits derived statistics to a public repository after each gameweek.
