"""
evaluate.py — Stage 5: Evaluation
====================================
Goes beyond a single MAE/R^2 number: plots predicted vs actual, looks at
residuals (where does the model do well/badly?), and writes a short plain-
language verdict — the "why it succeeds or fails" the project brief asks for.

Run after models.py (uses data/processed/model_predictions_sample.csv).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load(path=ROOT / "data" / "processed" / "model_predictions_sample.csv"):
    return pd.read_csv(path)


def predicted_vs_actual_plot(df: pd.DataFrame, model_col: str, label: str):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(df["finish_position"], df[model_col], alpha=0.75, s=60,
               color="#3671C6", edgecolor="black", linewidth=0.5)
    lims = [0, max(df["finish_position"].max(), df[model_col].max()) + 1]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, label="Perfect prediction")
    ax.set_xlabel("Actual finish position")
    ax.set_ylabel("Predicted finish position")
    ax.set_title(f"{label}: predicted vs actual\n(sample, n={len(df)})")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fname = f"pred_vs_actual_{model_col}.png"
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    return fname


def residual_plot(df: pd.DataFrame, model_col: str, label: str):
    df = df.copy()
    df["residual"] = df[model_col] - df["finish_position"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["grid_position"], df["residual"], alpha=0.75, s=60,
               color="#E8002D", edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Grid position")
    ax.set_ylabel("Residual (predicted - actual)")
    ax.set_title(f"{label}: residuals vs grid position\n"
                 f"(points above 0 = model predicted worse finish than actual)")
    fig.tight_layout()
    fname = f"residuals_{model_col}.png"
    fig.savefig(FIG_DIR / fname, dpi=150)
    plt.close(fig)
    return fname, df["residual"]


def biggest_misses(df: pd.DataFrame, model_col: str, n=5):
    df = df.copy()
    df["abs_error"] = (df[model_col] - df["finish_position"]).abs()
    return df.sort_values("abs_error", ascending=False).head(n)[
        ["driver_code", "constructor_name", "grid_position",
         "finish_position", model_col, "abs_error"]
    ]


def run():
    df = load()
    print("=" * 70)
    print(f"EVALUATION — n={len(df)} rows (sample dataset)")
    print("=" * 70)

    models = {
        "pred_baseline": "Baseline (grid=finish)",
        "pred_linear": "Linear Regression",
        "pred_random_forest": "Random Forest",
    }

    for col, label in models.items():
        fname1 = predicted_vs_actual_plot(df, col, label)
        fname2, residuals = residual_plot(df, col, label)
        print(f"\n{label}")
        print(f"  figures: {fname1}, {fname2}")
        print(f"  mean residual: {residuals.mean():+.2f}  "
              f"(positive = model tends to predict a worse finish than reality)")
        print(f"  Biggest individual misses:")
        misses = biggest_misses(df, col)
        print(misses.to_string(index=False))

    print("\n" + "-" * 70)
    print("WHY THE MODEL SUCCEEDS OR FAILS (plain-language verdict)")
    print("-" * 70)
    print("""
On this 2-race sample:
  - The baseline (just using grid position) already captures the dominant
    signal in F1 results: track position matters enormously, and a model
    needs a LOT of additional signal to beat "assume everyone stays where
    they started."
  - Random Forest and Linear Regression both did WORSE than the baseline
    here. With only 30 training examples spread over 5 cross-validation
    folds (~6 rows per fold), that is expected: there simply isn't enough
    data for the models to learn real patterns instead of noise, and
    OneHotEncoding the constructor column alone uses up a lot of that
    fold's information budget.
  - This is a genuine, defensible finding for this sample: "on 2 races of
    data, we can't yet show a learned model adds value over the naive
    baseline." That is exactly the kind of honest negative result a good
    data science report should state plainly, not hide.
  - The expected (not guaranteed) outcome once this is re-run on a full
    multi-season dataset (thousands of rows) is that Random Forest starts
    to beat the baseline, because it will have enough data to learn genuine,
    reusable patterns — e.g. how much certain circuits amplify overtaking,
    or how grid position interacts with tyre strategy — rather than just
    memorising noise in 30 rows.
""")


if __name__ == "__main__":
    run()
