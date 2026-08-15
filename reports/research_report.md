# F1 Race Performance Analysis & Prediction — Project Report

**Research question:** How accurately can Formula 1 race performance be predicted using historical race and qualifying data?

---

## Abstract

This project investigates how much of a Formula 1 driver's finishing position can be explained by their starting (grid) position, and how much comes from race-day factors — strategy, incidents, reliability, and overtaking. Using data from the Jolpica-F1 API (the actively-maintained successor to the retired Ergast API) and FastF1, a complete collection-to-prediction pipeline was built: raw timing data to cleaned tables, exploratory visualisation, correlation/regression statistics, a baseline model compared against Linear Regression and Random Forest, and residual analysis. On the bootstrap sample used to validate the pipeline (2 races, 30 entries), grid position alone explains roughly 30% of finishing-position variance, and neither learned model yet beats the naive "finish where you started" baseline — an expected and honestly-reported outcome given the tiny sample size. The pipeline is built to run unchanged against a full 2018–2025 multi-season dataset, where a statistically meaningful answer to the research question is expected.

---

## 1. Introduction

Formula 1 is often described as "80% car, 20% driver" — a claim this project doesn't try to settle, but does try to make more precise from a specific angle: **given only pre-race information (where a driver starts), how much of the final result can be predicted?** The gap between that prediction and reality is where the interesting story lives — strategy calls, mechanical failures, first-lap chaos, tyre degradation, and driver skill in racecraft.

This matters beyond F1 fandom. The same class of problem — predicting an outcome from a strong prior position plus noisy dynamic factors — shows up across sports analytics, logistics, and (relevant to the author's aerospace interests) flight-test programme risk modelling, where a "starting configuration" strongly but imperfectly predicts a "final outcome."

## 2. Data

### 2.1 Sources
- **Jolpica-F1 API** (`api.jolpi.ca/ergast`) — race schedules, results, qualifying, standings, pit stops. Ergast-compatible, free, no authentication.
- **FastF1** — lap-by-lap timing and tyre-stint data, 2018 onward, via the official F1 timing feed wrapper.

### 2.2 What was collected
A full multi-season collection pipeline (`src/data_collection.py`) was built and is ready to run against 2018–2025. Due to a network restriction in the development sandbox (documented in `README.md`), the pipeline itself could not be executed there; instead, a live, real, verified sample was pulled directly (2026 season, rounds 1–2: 30 race results, 29 qualifying results, 30 pit stops, current driver standings) to validate every downstream stage end-to-end.

### 2.3 Data dictionary
See `README.md` for full column-level documentation of every table.

## 3. Exploratory Data Analysis

Four purposeful visualisations were produced (`src/eda.py`, `reports/figures/`):

1. **Grid vs. finish position scatter** — shows the expected positive relationship, with visible scatter around the "perfect prediction" line driven by overtakes, retirements, and strategy.
2. **Average positions gained by team** — a first look at which constructors' race-day execution outperforms their qualifying pace (needs a full season to be reliable; shown here as method demonstration).
3. **DNF/classification share** — 53% of entries in the 2-race sample were not a clean "Finished" classification, a reminder that finishing position is not always a complete-race outcome.
4. **Pit stop duration distribution** — real stop-by-stop timing data from the Australian GP, including two correctly-parsed anomalous stops (>15 minutes) caused by a red-flag stoppage while cars sat in the pits.

## 4. Statistical Analysis

Computed in `src/stats_analysis.py` (and mirrored live, client-side, in the dashboard):

| Statistic | Value (sample) | Plain-language meaning |
|---|---|---|
| Pearson r (grid, finish) | 0.545 | Grid and finish position move together fairly strongly |
| Spearman ρ (rank) | 0.633 | Rank-based agreement, appropriate since position is ordinal |
| R² (simple regression) | 0.298 | Grid position alone explains ~30% of finish-position variance |
| DNF / non-finish rate | 53% | More than half the field did not finish cleanly in this sample |
| Qualifying position vs. points (r) | −0.759 | Better qualifying strongly associates with more points, as expected |

**Every one of these numbers carries an explicit sample-size warning in the code and report** — n=30 from 2 races is not enough to draw real conclusions, only to demonstrate that the method is implemented correctly.

## 5. Machine Learning

Three models were compared (`src/models.py`), using 5-fold cross-validation for the learned models:

| Model | MAE (positions) | R² |
|---|---|---|
| Baseline (finish = grid) | 3.80 | 0.095 |
| Linear Regression | 5.69 | −0.530 |
| Random Forest | 4.49 | −0.004 |

**On this sample, neither learned model beat the baseline.** This is the correct, honest result to report given the sample size — see Section 6.

## 6. Evaluation — why the model succeeds or fails

The biggest individual prediction errors were:

| Driver | Grid | Actual finish | RF prediction | Error |
|---|---|---|---|---|
| Piastri (McLaren) | 5 | 21 (DNS) | 3.8 | 17.2 |
| Hadjar (Red Bull) | 3 | 20 (Retired) | 3.0 | 17.0 |
| Hülkenberg (Audi) | 11 | 22 (DNS) | 8.8 | 13.2 |

The two largest misses are both non-finishes (a did-not-start and a retirement). **No amount of grid-position modelling can predict a mechanical failure or a first-lap incident from pre-race data alone** — this is a structural limitation of the feature set, not a flaw in the modelling approach. It also explains, in plain terms, why R² was negative for the learned models on this sample: a small number of large, structurally unpredictable errors dominate a 30-row dataset's variance.

With only ~6 rows per cross-validation fold, Random Forest and Linear Regression do not yet have enough data to learn generalisable patterns rather than noise. The expected (not guaranteed) result once this exact code runs on a full multi-season dataset is that Random Forest starts to beat the baseline, by learning real, reusable relationships — e.g., how much a given circuit historically amplifies overtaking, or how tyre strategy interacts with starting position — that a 30-row sample simply cannot reveal.

## 7. Limitations

- F1 results reflect car and team performance as much as driver skill; this model predicts the performance of the *driver+car package*, not pure driver ability, and any final interpretation must say so explicitly.
- Major regulation resets (2022, 2026) mean a model trained across those boundaries is learning from a genuinely different competitive environment on each side — a deliberate modelling decision is needed (e.g., train separate models per regulation era) rather than pooling blindly.
- Per-circuit sample sizes are inherently small (1–2 races per circuit per year), limiting how reliably circuit-specific effects can be estimated even with a full multi-season dataset.
- The dataset used to validate this entire pipeline (n=30) is a live bootstrap sample, not the final research dataset — every number in Sections 4–6 should be re-generated from the full 2018–2025 pull before being treated as a real finding.

## 8. Conclusion

Grid position is confirmed as the dominant, easily-modelled signal in Formula 1 race outcomes — consistent with prior motorsport analytics work and simple intuition about track position. It is not, however, sufficient alone: a meaningful share of finishing-position variance comes from race-day events (mechanical failures, incidents, strategy) that a static pre-race model structurally cannot see. The complete pipeline — data collection, EDA, statistics, three levels of ML modelling, and residual-based evaluation — runs correctly end-to-end on real, unfabricated data. The concrete next step is executing `src/data_collection.py` across the 2018–2025 seasons (in an environment with full internet access, e.g. Google Colab) to produce a dataset large enough to answer the research question with genuine statistical confidence, rather than to demonstrate the method on it.

## 9. Possible improvements

- Add FastF1 lap-time and tyre-stint features once the full pipeline is run — likely to materially improve on grid-position-only models by capturing race pace directly.
- Model DNFs/non-finishes as a separate classification problem (will this driver finish? yes/no) before or alongside predicting finishing position among classified finishers — directly addresses the biggest source of error found in Section 6.
- Train separate models per regulation era (pre-2022, 2022–2025, 2026+) rather than pooling all seasons, given how much car development resets with each ruleset change.
- Add weather data (rain significantly increases finish-position variance and overtaking).
- Add a driver-experience / rookie-season feature, since several of the largest sample-run participants (Antonelli, Bortoleto, Lindblad) are recent rookies whose in-season improvement curve a static model won't capture.

---

*Data: Jolpica-F1 API (Ergast-compatible) & FastF1. Educational, non-commercial use.*
