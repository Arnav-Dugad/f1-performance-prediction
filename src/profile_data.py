"""
profile_data.py
----------------
Stage 1 deliverable: "understand" the dataset, not just collect it.

Run this after data_collection.py (or on the bootstrap sample data) to print
a quick structural profile of each table: shape, dtypes, missing values, and
a few sanity checks specific to F1 data (e.g. grid position range, whether
finish position is ever missing for DNFs).

This has no effect on the data itself — it's read-only reporting.
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def profile(name: str, df: pd.DataFrame):
    print(f"\n{'=' * 60}\n{name}  —  shape={df.shape}\n{'=' * 60}")
    print(df.dtypes)
    print("\nMissing values per column:")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.sum() else "  (none)")
    print("\nFirst 3 rows:")
    print(df.head(3).to_string())


def main():
    processed = ROOT / "data" / "processed"
    files = {
        "2026 Race Schedule (sample)": processed / "2026_race_schedule.csv",
        "2026 Race Results (sample, rounds 1-2)": processed / "2026_race_results_sample.csv",
        "2026 Qualifying (sample, rounds 1-2)": processed / "2026_qualifying_sample.csv",
    }
    for name, path in files.items():
        if not path.exists():
            print(f"[skip] {path} not found — run parse_sample_data.py first")
            continue
        df = pd.read_csv(path)
        profile(name, df)

    # A couple of F1-specific sanity checks on the results table
    results_path = processed / "2026_race_results_sample.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        print(f"\n{'=' * 60}\nF1-specific sanity checks\n{'=' * 60}")
        print(f"Grid position range: {df.grid_position.min()}–{df.grid_position.max()}")
        print(f"Finish position range: {df.finish_position.min()}–{df.finish_position.max()}")
        dnf_count = (df.status != "Finished").sum()
        print(f"Non-'Finished' status rows (DNF/DNS/Lapped etc.): {dnf_count} / {len(df)}")
        corr = df["grid_position"].corr(df["finish_position"])
        print(f"Correlation(grid_position, finish_position) on this small sample: {corr:.3f}")


if __name__ == "__main__":
    main()
