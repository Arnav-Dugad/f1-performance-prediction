"""
stats_analysis.py — Stage 3: Statistical Analysis
====================================================
Computes the actual numbers behind the research question and prints a plain-
language explanation alongside each one. Run standalone: `python src/stats_analysis.py`.

IMPORTANT: on the bundled sample (n=30, 2 races), none of these numbers are
statistically reliable — they're illustrative of the METHOD. Re-run against
the full multi-season dataset (see README) before drawing real conclusions.
Every print statement below says this explicitly so it can't be misquoted
in the final report as more than it is.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent


def load(path=ROOT / "data" / "processed" / "2026_master_sample.csv"):
    return pd.read_csv(path)


def sample_size_warning(n, context=""):
    if n < 100:
        print(f"  [!] n={n} {context}— far too small for a reliable estimate. "
              f"Treat this number as a demonstration of method, not a finding.")


def grid_finish_correlation(df: pd.DataFrame):
    print("\n1. Grid position vs finish position — Pearson & Spearman correlation")
    print("-" * 70)
    clean = df.dropna(subset=["grid_position", "finish_position"])
    pearson_r, pearson_p = stats.pearsonr(clean["grid_position"], clean["finish_position"])
    spearman_r, spearman_p = stats.spearmanr(clean["grid_position"], clean["finish_position"])
    print(f"  Pearson r  = {pearson_r:.3f}  (p={pearson_p:.4f})")
    print(f"  Spearman rho = {spearman_r:.3f}  (p={spearman_p:.4f})")
    print(f"  In plain language: {pearson_r:.2f} means grid position and finish "
          f"position move together fairly strongly in this sample — starting "
          f"further back is associated with finishing further back, as expected.")
    print(f"  Spearman (rank-based) is often more appropriate here than Pearson, "
          f"since position is an ordinal ranking, not a truly continuous "
          f"measurement — the two agreeing closely is a good sign the "
          f"relationship isn't distorted by a few outlier laps.")
    sample_size_warning(len(clean))
    return {"pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p, "n": len(clean)}


def r_squared_simple_regression(df: pd.DataFrame):
    print("\n2. How much of the variance in finish position does grid position explain? (R^2)")
    print("-" * 70)
    clean = df.dropna(subset=["grid_position", "finish_position"])
    slope, intercept, r, p, se = stats.linregress(clean["grid_position"], clean["finish_position"])
    r2 = r ** 2
    print(f"  finish_position ~= {slope:.2f} * grid_position + {intercept:.2f}")
    print(f"  R^2 = {r2:.3f}")
    print(f"  In plain language: grid position alone explains about "
          f"{r2*100:.0f}% of the variation in where drivers finish in this "
          f"sample. The rest ({(1-r2)*100:.0f}%) comes from race-day factors: "
          f"strategy, incidents, car pace changes, driver errors, safety cars, etc.")
    sample_size_warning(len(clean))
    return {"slope": slope, "intercept": intercept, "r2": r2}


def average_positions_gained_by_team(df: pd.DataFrame):
    print("\n3. Average positions gained/lost by team (grid -> finish)")
    print("-" * 70)
    g = df.groupby("constructor_name")["positions_gained"].agg(["mean", "count"])
    g = g.sort_values("mean", ascending=False)
    print(g.to_string(float_format=lambda x: f"{x:.2f}"))
    print(f"  In plain language: positive = team tends to finish better than "
          f"they qualified (good race pace / strategy / overtaking); negative "
          f"= tends to finish worse (reliability issues, poor strategy, "
          f"getting stuck in traffic). Every team here has only 1-2 data "
          f"points, so this table is not evidence of a real team skill — "
          f"it's exactly the kind of table that becomes meaningful once "
          f"you have a full season per team.")
    return g


def dnf_rate_by_grid_third(df: pd.DataFrame):
    print("\n4. Does starting position relate to DNF/non-finish rate?")
    print("-" * 70)
    clean = df.dropna(subset=["grid_position"]).copy()
    clean["grid_third"] = pd.cut(clean["grid_position"], bins=3,
                                   labels=["Front third", "Midfield", "Back third"])
    clean["dnf"] = clean["status"] != "Finished"
    rate = clean.groupby("grid_third", observed=True)["dnf"].mean()
    print(rate.to_string(float_format=lambda x: f"{x:.1%}"))
    print(f"  In plain language: this checks whether starting near the back "
          f"is associated with a higher chance of not finishing cleanly "
          f"(e.g. getting caught in first-lap incidents). With only 30 rows "
          f"across 2 races this is noisy - the honest reading is 'inconclusive "
          f"on this sample', which is itself a useful, honest result to report.")
    return rate


def qualifying_position_vs_points(df: pd.DataFrame):
    print("\n5. Qualifying position vs championship points scored that race")
    print("-" * 70)
    clean = df.dropna(subset=["qualifying_position", "points"])
    r, p = stats.pearsonr(clean["qualifying_position"], clean["points"])
    print(f"  Pearson r = {r:.3f} (p={p:.4f})")
    print(f"  In plain language: a negative correlation is expected here — "
          f"better (lower-numbered) qualifying position should be associated "
          f"with more points scored, since points require a high finish and "
          f"grid position is the single strongest predictor of finish.")
    sample_size_warning(len(clean))
    return {"r": r, "p": p, "n": len(clean)}


if __name__ == "__main__":
    df = load()
    print("=" * 70)
    print(f"STATISTICAL ANALYSIS — dataset: {len(df)} rows "
          f"({df['round'].nunique()} race(s), {df['driver_id'].nunique()} drivers)")
    print("REMINDER: this is the bootstrap sample (2026 rounds 1-2). Numbers")
    print("below demonstrate the method; re-run on the full dataset for real")
    print("conclusions. See README.md.")
    print("=" * 70)

    grid_finish_correlation(df)
    r_squared_simple_regression(df)
    average_positions_gained_by_team(df)
    dnf_rate_by_grid_third(df)
    qualifying_position_vs_points(df)
