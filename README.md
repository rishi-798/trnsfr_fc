# transfer fc

transfer fc is a project that looks at premier league player performance and estimated market values from 2022/23 to the current 2026/27 season.

it uses transfermarkt data to build player-season records, then adds current-season stats from the fantasy premier league (fpl) feed after each gameweek. a python pipeline trains a model to estimate player values, and the results are displayed on a static website.

## features

- browse players by season and position

- sort players by market value or performance score

- search for players or clubs

- track the 2026/27 season with updated estimates after every gameweek

- view player stats, value history, and model estimates

- see how different factors affect each player's estimated value

- compare the model against a baseline that keeps each player's starting value unchanged

- review data quality using a sample of up to 50 player-seasons

## running the site

if site/data/players.json already exists, you can run the website without installing the modeling dependencies.

from the project root, run:

```bash
python3 scripts/serve_site.py
```

then open http://localhost:8100.

the site needs to run over http because it loads data using fetch(). the server also tells the browser to check for updated files every time the page loads, so changes to the data and code show up without a hard refresh.

if port 8100 is already being used, you can pass a different port. press Ctrl+C to stop the server.

no node.js build step or backend is needed.

## setting up the python pipeline

the project uses python 3 with venv and pip.

dependencies are listed in requirements.txt and include pandas, numpy, scikit-learn, certifi, and jupyter. versions are currently unpinned.

clone the repository and install everything:

```bash
git clone https://github.com/rishi-798/trnsfr_fc.git
cd trnsfr_fc
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

on windows, use .venv\Scripts\activate instead.

if you already have the repository, start from the environment creation step.

## building the data and predictions

run these commands in order from the project root:

```bash
python scripts/download_data.py
python scripts/build_training_table.py
python scripts/update_live.py
python scripts/train_model.py
python scripts/export_site_data.py
python scripts/write_explanations.py  # optional
python scripts/export_site_data.py
```

update_live.py adds the current season using the public fpl feed. you can skip it if you only want to build the historical seasons.

write_explanations.py is optional and uses gemini to rewrite the model explanations. it requires a GEMINI_API_KEY.

each step creates the following files:

| step | output |
| --- | --- |
| download | six compressed csv files in data/raw/ |
| build training table | data/processed/player_seasons.csv |
| train and evaluate | predictions.csv, explanations.csv, and model_metrics.json |
| export | site/data/players.json |

the downloader needs internet access and skips files that already exist. to download a file again, remove or move the existing raw file before running the script.

the build, training, and export steps replace their previous outputs. if you change the raw data or feature logic, rerun the later steps to keep everything updated.

the exporter can also run before training. in that case, the site will still show observed values and performance scores, but no model predictions.

for exploratory analysis, run:

```bash
jupyter notebook notebooks/01_explore.ipynb
```

## data conventions

each row represents a player who made at least one premier league appearance in a season.

seasons use their starting year. for example, 2025 means 2025/26.

value_end is the latest available valuation on or before june 30 after the season ends.

historical values use the same june 30 cutoff one, two, and three years earlier. all values are in euros.

a row is marked usable_for_training only if it has an ending valuation that is no more than 180 days old at the cutoff. training also requires a starting valuation.

current-season league stats only include premier league appearances. other domestic league appearances are stored separately in other_league_* fields.

a player's main club is the club where they played the most premier league minutes that season.

previous-season totals combine all covered domestic leagues. prev_league identifies the league where they played the most minutes, but the totals are not limited to that league.

players without a season-start valuation are left out of the site because there is no starting value or prediction to display. performance rankings only include players shown on the site.

missing history stays missing and includes reason columns. previous-season league stats are only set to zero when other recorded appearances confirm that the player participated in a covered league but had no recorded league minutes.

## model and evaluation

the model uses linear regression with a small ridge penalty. it combines StandardScaler and RidgeCV, which chooses the penalty through cross-validation.

the model uses:

- age and position

- playing time

- goals and assists

- club context

- previous-season performance

- valuation history

the target is the logarithmic change in player value.

some features are transformed before training:

- starting value is converted to a logarithm.

- value trends use log ratios between consecutive june 30 valuations.

- age includes a squared term centered on 26, allowing values to increase and then decrease with age.

- goals and assists are measured per 90 minutes.

minutes and goals are also adjusted based on how far the starting value is above €20m, using a log scale capped at €100m. this helps account for the fact that the same performance usually adds less value to an already expensive player.

missing history is filled with 0 and tracked using indicator columns.

the prediction works like this:

```text
target = log(value_end / value_1y_ago)
predicted_value = value_1y_ago * exp(predicted_log_change)
```

because the model uses completed-season performance, its estimates are retrospective season-end estimates. they are not preseason forecasts or predictions of actual transfer fees.

### evaluation results

the model trains on 2022/23 through 2024/25 and tests on 2025/26.

the saved results include 1,560 training rows and 506 test rows.

| metric on 2025/26 | model | unchanged-value baseline |
| --- | --- | --- |
| mean absolute log error (MALE) | 0.2723 | 0.4808 |
| mean absolute error (EUR) | €5,201,563 | €8,179,595 |

lower is better for both metrics.

these results are from the saved run. changing the data or dependencies can change them.

a 500-tree random forest scored 0.2640 MALE on the same test set. the linear model was chosen instead because it is easier to understand and explain.

to see errors by season, position, starting value, promoted clubs, and missing previous-season stats, run:

```bash
python scripts/evaluate_groups.py
```

## how site predictions work

the site uses walk-forward training. each season is predicted using only data from earlier seasons.

there are no predictions for 2022/23 because there is no earlier season to train on. players without a starting valuation also don't receive predictions.

the site's actual value is the player's market value at the start of the season. if a prediction is unavailable, the site falls back to that value.

each estimate also has an explanation showing how different features contributed to the predicted change in value. these contributions are exact for the linear model and are calculated using the model weight multiplied by the player's standardized distance from the training average.

the contributions add up to the prediction, but they describe how the model works, not what actually causes a player's value to change.

## live 2026/27 season

the transfermarkt dataset currently ends with 2025/26, so the current season is added separately using the public fantasy premier league api.

the historical dataset already contains everything needed from before the current season:

- june 30, 2026 starting values

- previous valuation history

- 2025/26 statistics

scripts/update_live.py adds current-season squads, positions, minutes, goals, assists, cards, clean sheets, and league table data from fpl.

### how live updates work

the script only rebuilds the data when fpl marks a new gameweek as final, meaning the games are finished and the data has been checked.

it adds up each player's stats across all finished gameweeks so appearances and totals stay consistent.

- --check checks whether a new gameweek is available.

- --force rebuilds the data even if there isn't a new gameweek.

fpl players are matched to transfermarkt using birth dates and shared name words. if that doesn't work, the script tries close name matches with birth dates within two days, including swapped day and month values.

players who are difficult to match can be added manually to data/live/fpl_matches.csv using fpl_code and player_id. every match should be checked in data/live/matches.csv.

players missing from the transfermarkt dataset, mostly championship-only players and teenagers, don't have a starting valuation and are left out.

### current-season projections

the model was trained on full seasons, so current-season stats are projected to a full 38-game season.

the projection uses:

- 60% of the player's current-season pace

- 40% of their previous-season totals

this helps prevent a strong or weak start from having too much influence. players without previous-season stats only use their current-season pace.

all features are also capped at the ranges seen in training so early-season projections don't go beyond what the model has seen before.

early estimates are still noisy and should become more stable as the season progresses.

champions league participation is not included in the fpl feed. the five english clubs in the 2026/27 league phase are listed in UCL_STAGE inside update_live.py. their stage can be updated as the competition progresses.

there are no in-season valuations, and the actual season-end values won't be available until june 2027. because of this, the current-season forecasts can't be evaluated yet.

### automatic updates

a github actions workflow checks for new gameweeks every day at 07:00 UTC.

when a new gameweek is final, it automatically:

- updates the live data

- retrains the model

- exports the updated site data

- commits the changes to main

if you're working locally, pull the latest changes before starting.

you can also run the workflow manually from the repository's actions tab, with an option to force an update.

## gemini explanations

scripts/write_explanations.py uses gemini to rewrite each model explanation into two simple sentences.

it uses gemini-3.8-flash and falls back to gemini-3.6-flash if the first model is overloaded. if both are unavailable, it waits and retries. if it still can't get a response, it keeps the original template.

the script runs before the site is updated and never calls gemini directly from the browser.

gemini only receives the information behind each explanation:

- the original template sentence

- the model's factor contributions

- key player statistics

every rewrite is checked before being accepted. it gets rejected if it:

- uses numbers that aren't in the provided data

- mentions things the model doesn't know about, like injuries, contracts, transfer fees, rumors, managers, or international football

- doesn't explain whether the estimated value increases or decreases

- isn't exactly two sentences or exceeds 70 words

- leaves out the predicted value

if a rewrite is rejected, the original template stays.

rewrites are saved in data/processed/explanations_ai.csv along with a fingerprint of the facts, model, and prompt version.

the exporter only uses a rewrite if it still matches the current data. only players whose facts have changed are sent to gemini again.

the current season is processed first, so if the daily quota runs out, you can rerun the script later to continue.

on the site, the rewritten explanation is shown with the original template underneath as "details".

to enable gemini, add this to a .env file in the project root:

```bash
GEMINI_API_KEY=...
```

the .env file is ignored by git.

for automatic weekly updates, add the same key as a github repository secret named GEMINI_API_KEY under Settings → Secrets and variables → Actions.

without the key, new gameweeks will still update, but the explanations will use the original templates.

## performance score

the performance score is separate from the valuation model. it's a custom score used to rank players based on their position-specific stats per 90 minutes.

| position | positive contributions |
| --- | --- |
| forwards | 5 × goals + 4 × assists |
| midfielders | 3 × goals + 4 × assists |
| defenders | 3 × clean sheets + 3 × goals + 3 × assists |
| goalkeepers | 5 × clean sheets |

all positions also subtract:

- 2 × yellow cards

- 5 × red cards

these are also calculated per 90 minutes.

the final score is multiplied by:

```text
minutes / (minutes + 900)
```

this reduces the impact of players with very little playing time.

a clean sheet requires at least 60 minutes played and zero goals conceded by the team.

rankings and percentiles compare players within the same season and position.

## data review and limitations

after building the training table, you can create a sample for manual review:

```bash
python scripts/make_pilot_sample.py
```

this creates data/processed/pilot_sample.csv with review_status and review_notes columns.

the script won't overwrite an existing sample that has review entries. make sure to preserve that file before generating another one. using --force will discard existing review entries.

the existing pilot review contains 20 reviewed rows, including three valuation discrepancies. these reviews did not change the underlying labels. the sample was deliberately selected and should not be treated as an estimate of the error rate across the full dataset.

there are also some limitations to keep in mind:

- league coverage is incomplete.

- valuation coverage is incomplete.

- missing history doesn't always mean a player didn't play.

the local source snapshot used in the exploration notebook has a valuation cutoff of june 12, 2026.

these gaps should be considered when interpreting player estimates and performance scores.

## project structure

```text
data/
  raw/                         downloaded source tables
  processed/                   training data, predictions, metrics, and reviews
  live/                        live-season data and manual player matches

notebooks/
  01_explore.ipynb              initial exploration and coverage checks

scripts/
  download_data.py              download source tables
  build_training_table.py       build player-season records
  make_pilot_sample.py          generate a manual review sample
  train_model.py                train and evaluate the model
  update_live.py                add current-season fpl data
  write_explanations.py         rewrite model explanations with gemini
  export_site_data.py           export data for the website
  evaluate_groups.py             compare model errors across groups
  test_staleness.py              test different valuation staleness cutoffs
  serve_site.py                  run the site locally

site/
  index.html                     website entry point
  style.css                      website styling
  app.js                         filtering, sorting, and player details
  data/players.json              exported player data

.github/workflows/
  refresh-live.yml               daily live-season update workflow

requirements.txt                 python dependencies
```

## data sources

historical data comes from the public dcaribou/transfermarkt-datasets project, which is derived from transfermarkt. check the upstream repository for its documentation and licensing terms.

current 2026/27 stats come from the public fantasy premier league api:

fantasy.premierleague.com/api

the api is unofficial and undocumented, and its terms for republishing data have not been confirmed. this project currently commits derived statistics to a public github repository after each gameweek.
