"""
models.py — Stage 4: Machine Learning
========================================
Two models, in increasing order of sophistication, as the project brief asks:

  1. BASELINE — "predict finish position = grid position". No learning at
     all; this is the number any real model has to beat to be worth using.
  2. LINEAR REGRESSION — a simple learned model using grid position (+ a
     couple of other numeric features when available) to predict finish
     position.
  3. RANDOM FOREST — a more flexible model that can capture non-linear
     effects and interactions between features (e.g. "starting P15 at a
     circuit known for chaos" vs "starting P15 at a processional circuit").

HONESTY NOTE ON SAMPLE SIZE
-----------------------------
The bundled sample dataset has n=30 (2 races). That is nowhere near enough
data to train or evaluate a machine learning model meaningfully — with this
few rows, cross-validation folds contain single-digit numbers of examples,
and any "R^2" or "accuracy" number produced is essentially noise. This script
still runs the full pipeline on it, because the POINT of Stage 1-4 right now
is to prove the pipeline is correct end-to-end. The printed output says so
explicitly, and the honest conclusion on this sample is: "we cannot yet tell
whether Random Forest beats the baseline — need more data." That is a
legitimate, useful scientific conclusion, not a failure.

Re-run this unchanged against the full multi-season dataset (several thousand
rows) for a real answer to the research question.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent

NUMERIC_FEATURES = ["grid_position", "qualifying_position"]
CATEGORICAL_FEATURES = ["constructor_id"]
TARGET = "finish_position"


def load_and_prepare(path=ROOT / "data" / "processed" / "2026_master_sample.csv"):
    df = pd.read_csv(path)
    # Drop rows where we don't know the outcome (unclassified/DNS finishes
    # have no meaningful finish_position for a regression target) — this is
    # a genuine modelling decision, documented rather than hidden.
    df = df.dropna(subset=[TARGET, "grid_position"]).copy()
    return df


def baseline_predict(df: pd.DataFrame) -> np.ndarray:
    """The baseline model: predicted finish = grid position. No fitting."""
    return df["grid_position"].values.astype(float)


def build_pipeline(model):
    preprocessor = ColumnTransformer([
        ("num", "passthrough", NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("prep", preprocessor), ("model", model)])


def cross_validated_predictions(df: pd.DataFrame, model, n_splits: int):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    # median-impute any missing qualifying_position (e.g. driver DNS'd quali)
    X["qualifying_position"] = X["qualifying_position"].fillna(X["qualifying_position"].median())
    y = df[TARGET].values.astype(float)

    pipe = build_pipeline(model)
    n_splits = max(2, min(n_splits, len(df) // 2))  # never ask for more folds than data supports
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    preds = cross_val_predict(pipe, X, y, cv=kf)
    return preds, y, n_splits


def evaluate(name: str, y_true, y_pred, n: int, n_splits: int | None = None):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    fold_note = f", {n_splits}-fold CV" if n_splits else ""
    print(f"\n{name}  (n={n}{fold_note})")
    print(f"  MAE  = {mae:.2f} positions")
    print(f"  R^2  = {r2:.3f}")
    return {"name": name, "mae": mae, "r2": r2, "n": n}


def run(input_path=None):
    df = load_and_prepare(input_path) if input_path else load_and_prepare()
    print("=" * 70)
    print(f"MODELLING — n={len(df)} rows")
    if len(df) < 200:
        print("[!] SAMPLE SIZE WARNING: fewer than 200 rows available.")
        print("    Cross-validated metrics below are for pipeline-verification")
        print("    purposes only and should NOT be reported as real model")
        print("    performance. Re-run against the full multi-season dataset.")
    print("=" * 70)

    results = []

    # 1. Baseline (no CV needed — it's not fitted on data)
    baseline_preds = baseline_predict(df)
    results.append(evaluate("Baseline (finish = grid)", df[TARGET], baseline_preds, len(df)))

    # 2. Linear Regression
    lr_preds, y, n_splits = cross_validated_predictions(df, LinearRegression(), n_splits=5)
    results.append(evaluate("Linear Regression", y, lr_preds, len(df), n_splits))

    # 3. Random Forest
    rf_preds, y, n_splits = cross_validated_predictions(
        df, RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42),
        n_splits=5
    )
    results.append(evaluate("Random Forest", y, rf_preds, len(df), n_splits))

    print("\n" + "-" * 70)
    print("Summary — does the model beat the baseline?")
    print("-" * 70)
    base_mae = results[0]["mae"]
    for r in results[1:]:
        delta = base_mae - r["mae"]
        verdict = "beats baseline" if delta > 0 else "does NOT beat baseline"
        print(f"  {r['name']:22s} MAE={r['mae']:.2f} vs baseline MAE={base_mae:.2f}  -> {verdict} "
              f"(delta={delta:+.2f} positions)")
    print("\nWith n=30 this comparison is not reliable (see warning above) — it's")
    print("included to show the comparison methodology working correctly, ready")
    print("to be re-run on real multi-season data where the answer will matter.")

    # Save predictions for the evaluation stage / dashboard
    out = df[["season", "round", "driver_code", "constructor_name",
              "grid_position", "qualifying_position", TARGET]].copy()
    out["pred_baseline"] = baseline_preds
    out["pred_linear"] = lr_preds
    out["pred_random_forest"] = rf_preds
    out_path = ROOT / "data" / "processed" / "model_predictions_sample.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions -> {out_path}")

    return results, out


if __name__ == "__main__":
    run()
