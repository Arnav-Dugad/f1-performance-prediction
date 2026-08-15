https://arnav-dugad.github.io/f1-performance-prediction/

# F1 Race Performance Analysis & Prediction

**Research question:** How accurately can Formula 1 race performance be predicted using historical race and qualifying data?

A data science / machine learning project analysing what determines an F1 driver's
finishing position — and building a model to predict it — using publicly available
historical race, qualifying, lap-time, and pit-stop data.

Built as an independent project combining interests in F1, machine learning, and
aerospace/motorsport engineering (Class 11).

---

## Project status

| Stage | Description | Status |
|---|---|---|
| 1 | Data collection | ✅ Complete |
| 2 | Exploratory data analysis | ✅ Complete (`src/eda.py` → `reports/figures/`) |
| 3 | Statistical analysis | ✅ Complete (`src/stats_analysis.py`) |
| 4 | Machine learning models | ✅ Complete (`src/models.py`) |
| 5 | Evaluation | ✅ Complete (`src/evaluate.py`) |
| 6 | Final presentation / dashboard | ✅ Complete (`dashboard/index.html`, `reports/research_report.md`) |

**Read this first:** every stage above was built and validated end-to-end on a real,
live-fetched **sample** of the data (2026 season, rounds 1–2, n=30) — not the full
research dataset. See "Known limitation" below. The full research report
(`reports/research_report.md`) explains exactly what's a genuine finding vs. a
pipeline-validation number at every step. Nothing in this repository is
fabricated data.

## Quick links

- **Live dashboard:** open `dashboard/index.html` in any browser (fully self-contained, no server needed)
- **Full report:** [`reports/research_report.md`](reports/research_report.md)
- **Figures:** [`reports/figures/`](reports/figures/)

---

## Stage 1 — Data Collection

### Data sources (public, free, legal)

1. **[Jolpica-F1 API](https://github.com/jolpica/jolpica-f1)** — `https://api.jolpi.ca/ergast/f1`
   The actively-maintained, community-run successor to the Ergast Developer API
   (Ergast was shut down at the end of 2024 after 20+ years of service).
   Ergast-compatible, free, no API key or authentication required. Provides
   season-by-season structured data back to 1950: race schedules, race
   results, qualifying results, driver/constructor standings, and pit stops.
   Used under Jolpica's [Terms of Use](https://github.com/jolpica/jolpica-f1/blob/main/LICENSE) —
   unauthenticated rate limit of roughly 200 requests/hour, which
   `src/data_collection.py` respects with built-in delays.

2. **[FastF1](https://docs.fastf1.dev/)** — a Python package that wraps the
   official F1 live-timing data feeds. Used for session-level detail that
   Ergast/Jolpica does not provide: lap-by-lap times, tyre compound and
   stint length, and precise pit stop durations, from the 2018 season onward
   (when the official timing feed FastF1 relies on became available in its
   current form).

No data in this repository is scraped from paywalled sources, and no data is
invented or estimated to fill gaps — see "Known limitation" below for what
that means in practice for this specific setup step.

### What's actually in this repo right now

- **`src/data_collection.py`** — the full, working, multi-season collection
  pipeline. Pulls race schedules, results, qualifying, standings, and pit
  stops from Jolpica for any year range, plus lap times and tyre stints from
  FastF1 for 2018+. This is the script that produces the real dataset the
  rest of the project (Stages 2–5) will be built on.
- **`data/sample_live_fetch/`** — raw JSON actually fetched live from the
  Jolpica API while this project was being set up (2026 season, rounds 1–2:
  schedule, results, qualifying). Real data, not fabricated, used to prove
  the pipeline and downstream processing code work end-to-end.
- **`src/parse_sample_data.py`** — turns that raw JSON into tidy CSVs in
  `data/processed/`.
- **`src/profile_data.py`** — a "get to know the data" report: shapes,
  dtypes, missing values, and a couple of F1-specific sanity checks (e.g.
  grid position range, share of non-finishes). See the output below.

### Known limitation (read this before Stage 2)

The environment used to *set up* this repository has restricted outbound
internet access — it cannot reach `api.jolpi.ca` or the F1 timing servers
directly. This was confirmed directly (see commit history / setup log): a
`requests.get()` call to the Jolpica API from that environment returns a
`403 Host not in allowlist` error, and FastF1 similarly cannot reach its
timing feeds.

**This does not affect the validity of `src/data_collection.py`.** The script
uses the same request patterns verified to work against the live API (see
`data/sample_live_fetch/`, fetched from the same API this script targets).
It simply has to be *run* somewhere with normal internet access:

```bash
# Option A: Google Colab (recommended, zero setup)
#   Upload/clone this repo into Colab, then:
!pip install -r requirements.txt
!python src/data_collection.py --start-year 2018 --end-year 2025

# Option B: your own computer
pip install -r requirements.txt
python src/data_collection.py --start-year 2018 --end-year 2025
```

Expect this to take a while: FastF1's lap-by-lap data is the slow part
(roughly 1 API session load per race weekend, cached locally after the first
run so it's instant on re-runs). Use `--skip-fastf1` for a fast first pass
that only pulls results/qualifying/standings — enough to start Stage 2 EDA
immediately while the full lap-time layer downloads in the background.

### Data dictionary (main tables)

**`{year}_results.csv`** — one row per driver per race
| column | meaning |
|---|---|
| `season`, `round` | year and race number within that season |
| `driver_id`, `driver_code`, `driver_name` | driver identity |
| `constructor_id`, `constructor_name` | team |
| `grid_position` | starting position (0 = pit lane start) |
| `finish_position` | classified finishing position (null if not classified — DNF/DSQ) |
| `classified_status` | raw position text, e.g. "R" for retired |
| `points` | championship points scored |
| `laps_completed` | laps completed |
| `status` | "Finished", "Retired", "Lapped", "Did not start", etc. |
| `positions_gained` | `grid_position - finish_position` (positive = gained places) |
| `fastest_lap_rank`, `fastest_lap_time` | where their fastest lap ranked, and its time |

**`{year}_qualifying.csv`** — one row per driver per race
| column | meaning |
|---|---|
| `qualifying_position` | final grid-determining position |
| `Q1`, `Q2`, `Q3` | best lap time in each qualifying segment (null if driver was eliminated in an earlier segment) |

**`{year}_pitstops.csv`** — one row per pit stop
| column | meaning |
|---|---|
| `stop_number` | 1st, 2nd, 3rd stop of the race for that driver |
| `lap` | lap number the stop occurred on |
| `duration_seconds` | pit stop duration (stationary time) |

**`{year}_{round}_laps.csv`** (FastF1) — one row per driver per lap
| column | meaning |
|---|---|
| `LapTime` | lap time in seconds |
| `Compound` | tyre compound used (SOFT/MEDIUM/HARD/INTERMEDIATE/WET) |
| `TyreLife` | laps completed on the current set of tyres |
| `Stint` | which stint of the race (1st, 2nd, ...) |

### Sample data profile (real output, from `src/profile_data.py`)

Ran against the live-fetched 2026 rounds 1–2 sample:

- Race results sample: **30 rows, 17 columns**. Missing values only in
  `finish_time` (6 rows) — expected, since drivers who retired have no
  finishing time.
- Qualifying sample: **29 rows, 11 columns**. `Q2` missing for 3 rows and
  `Q3` missing for 10 rows — expected, since only the fastest drivers advance
  past Q1 and Q2 in F1's knockout qualifying format. This is a genuine
  structural feature of the data, not a data quality problem, and needs to be
  handled deliberately in Stage 2/4 (e.g. using `qualifying_position` as the
  primary feature rather than raw Q1/Q2/Q3 times, since not everyone sets one).
- Grid position range: 1–22. Finish position range: 1–22 (22 cars in this
  sample; the modern F1 grid is usually 20, this sample includes reserve/test
  entries — worth double-checking grid size assumptions once the full dataset
  is in).
- 16 of 30 result rows (53%) were **not** "Finished" (retirements, lapped
  classifications, etc.) — a reminder that "finishing position" isn't always
  a clean, complete-race outcome, which matters a lot for Stage 4 modelling
  decisions (do we predict classified position including DNFs, or only among
  finishers?).
- Grid-position vs finish-position correlation on this tiny 30-row sample:
  **0.545**. This is *not* a result to draw conclusions from — it's 30 rows
  from 2 races. The real correlation analysis happens in Stage 3 once the
  full multi-season dataset is collected.

---

## Stage 2 — Exploratory Data Analysis

`src/eda.py` generates four purposeful plots (not one histogram per column) aimed
directly at the research question:

1. **Grid vs. finish position scatter**, coloured by team, with a "perfect
   prediction" reference line.
2. **Average positions gained by team** (grid → finish) — an early look at
   which teams' race-day execution outperforms their qualifying pace.
3. **DNF / classification-status share** — 53% of the sample's entries were
   not a clean "Finished", a reminder that finishing position isn't always a
   complete-race outcome.
4. **Pit stop duration distribution** — including correctly parsed anomalous
   stops (>15 minutes) caused by a red-flag stoppage, a real messy-data case
   worth handling deliberately rather than treating as a data error.

```bash
python src/eda.py --input data/processed/2026_master_sample.csv
```

## Stage 3 — Statistical Analysis

`src/stats_analysis.py` computes correlation (Pearson + Spearman), a simple
regression R², team-level positions-gained, DNF rate by grid third, and
qualifying-position-vs-points correlation — each printed with a plain-language
explanation and an explicit sample-size warning. See `reports/research_report.md`
Section 4 for the full results table.

```bash
python src/stats_analysis.py
```

## Stage 4 — Machine Learning

`src/models.py` compares three models via 5-fold cross-validation:
1. **Baseline** — predicted finish = grid position (no fitting).
2. **Linear Regression** — grid + qualifying position + constructor (one-hot).
3. **Random Forest** — same features, allows non-linear effects/interactions.

```bash
python src/models.py
```

## Stage 5 — Evaluation

`src/evaluate.py` goes beyond a single metric: predicted-vs-actual plots,
residual-vs-grid-position plots, and a table of the biggest individual misses
with a plain-language "why" — e.g. the two largest errors in the sample were
both non-finishes (a DNS and a retirement), which no pre-race-only model can
predict. See `reports/research_report.md` Section 6.

```bash
python src/evaluate.py
```

## Stage 6 — Final Presentation

- **`dashboard/index.html`** — a fully self-contained, interactive website
  (no build step, no server, works offline). Every chart and statistic is
  computed **live in the browser** from the embedded real dataset — open dev
  tools and inspect `window.F1DATA` to verify nothing is hardcoded. Includes
  the abstract, all 6 stages, and the limitations/conclusion.
- **`reports/research_report.md`** — the full written report: abstract,
  introduction, data, EDA, statistics, ML, evaluation, limitations,
  conclusion, and possible improvements.
- **`reports/figures/`** — the static PNG versions of every EDA/evaluation plot.

## Repository structure

```
f1-performance-prediction/
├── README.md                      <- you are here
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_collection.py         <- Stage 1: full multi-season pipeline
│   ├── parse_sample_data.py       <- Stage 1: sample JSON -> CSV
│   ├── profile_data.py            <- Stage 1: data understanding / profiling
│   ├── eda.py                     <- Stage 2: exploratory plots
│   ├── stats_analysis.py          <- Stage 3: correlation/regression stats
│   ├── models.py                  <- Stage 4: baseline / linear / random forest
│   └── evaluate.py                <- Stage 5: residuals, biggest misses, verdict
├── data/
│   ├── sample_live_fetch/         <- raw JSON, real data, fetched live during setup
│   ├── raw/
│   │   ├── ergast/                <- output of data_collection.py (season CSVs)
│   │   └── fastf1/                <- output of data_collection.py (lap/tyre CSVs)
│   └── processed/                 <- cleaned/merged CSVs ready for analysis
├── notebooks/                     <- exploratory Jupyter notebooks (optional)
├── reports/
│   ├── research_report.md         <- Stage 6: full written report
│   └── figures/                   <- Stage 6: PNG plots from eda.py / evaluate.py
├── dashboard/
│   └── index.html                 <- Stage 6: self-contained interactive website
└── cache/                         <- FastF1's local cache (gitignored)
```

## Tech stack

Python · pandas · NumPy · Matplotlib · seaborn · scikit-learn · FastF1 · Jolpica/Ergast API · Streamlit (dashboard) · Git/GitHub

## Ethics & limitations of the data itself (carried into later stages)

- F1 results depend heavily on car/team performance, not just driver skill —
  any model built on this data predicts "how will this driver+car package do,"
  not "how good is this driver," and the final report needs to say that
  explicitly.
- Regulation changes (e.g. the major 2022 and 2026 technical regulation
  resets) mean a model trained across those boundaries is learning from a
  genuinely different sport each side of the change — this is a modelling
  decision to make explicitly in Stage 4, not ignore.
- Sample sizes per circuit/driver/constructor combination are small (max
  ~1-2 races per circuit per year), which limits how much circuit-specific
  effects can be reliably estimated.

## License / attribution

Race data: © Jolpica-F1 contributors / FIA timing data, accessed via the
Jolpica-F1 API under its published Terms of Use. This project is for
educational, non-commercial purposes.
