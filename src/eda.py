"""
eda.py — Stage 2: Exploratory Data Analysis
=============================================
Generates a small set of purposeful plots (not "one histogram per column")
aimed directly at the research question: how does qualifying/grid position
relate to finishing position, and what else seems to matter?

Reads data/processed/2026_master_sample.csv by default. Point it at a fuller
multi-season file (once you've run data_collection.py) via --input to get the
real, statistically meaningful version of the same plots — the code doesn't
change, only the data does.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11,
})

TEAM_COLORS = {
    "mercedes": "#27F4D2", "ferrari": "#E8002D", "red_bull": "#3671C6",
    "mclaren": "#FF8000", "aston_martin": "#00665F", "alpine": "#FF87BC",
    "williams": "#64C4FF", "rb": "#6692FF", "haas": "#B6BABD",
    "audi": "#00302B", "cadillac": "#8E4F00",
}


def plot_grid_vs_finish(df: pd.DataFrame, suffix=""):
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = df["constructor_id"].map(TEAM_COLORS).fillna("#888888")
    ax.scatter(df["grid_position"], df["finish_position"], c=colors, s=70,
               edgecolor="black", linewidth=0.5, alpha=0.9, zorder=3)
    lims = [0, max(df["grid_position"].max(), df["finish_position"].max()) + 1]
    ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1,
             label="Perfect prediction (finish = grid)", zorder=1)
    ax.set_xlabel("Grid (starting) position")
    ax.set_ylabel("Finishing position")
    ax.set_title("Grid position vs finishing position" + suffix)
    ax.invert_yaxis(); ax.invert_xaxis()  # P1 top-right, reads naturally
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"grid_vs_finish{'_sample'}.png", dpi=150)
    plt.close(fig)


def plot_positions_gained_by_team(df: pd.DataFrame, suffix=""):
    order = (df.groupby("constructor_name")["positions_gained"]
             .mean().sort_values(ascending=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [TEAM_COLORS.get(cid, "#888888")
              for cid in df.groupby("constructor_name")["constructor_id"].first()[order.index]]
    ax.bar(order.index, order.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Average positions gained (grid to finish)")
    ax.set_title("Average positions gained by team" + suffix)
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"positions_gained_by_team{'_sample'}.png", dpi=150)
    plt.close(fig)


def plot_dnf_rate(df: pd.DataFrame, suffix=""):
    status_counts = df["status"].apply(
        lambda s: "Finished" if s == "Finished" else "Did not finish / classify"
    ).value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.0f%%",
           colors=["#3671C6", "#E8002D"], startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("Share of results that finished cleanly" + suffix)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"dnf_share{'_sample'}.png", dpi=150)
    plt.close(fig)


def plot_pitstop_durations(pitstops: pd.DataFrame):
    clean = pitstops[~pitstops["unusually_long_stop"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(clean["duration_seconds"], bins=15, color="#3671C6",
            edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pit stop duration (seconds)")
    ax.set_ylabel("Count")
    ax.set_title("Pit stop duration distribution\n(Australian GP 2026 - excludes 2 red-flag-affected stops)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pitstop_durations.png", dpi=150)
    plt.close(fig)


def run(input_path: Path):
    df = pd.read_csv(input_path)
    suffix = "\n(sample: 2026 rounds 1-2, n={})".format(len(df))
    plot_grid_vs_finish(df, suffix)
    plot_positions_gained_by_team(df, suffix)
    plot_dnf_rate(df, suffix)

    pitstop_path = ROOT / "data" / "processed" / "2026_r1_pitstops_sample.csv"
    if pitstop_path.exists():
        plot_pitstop_durations(pd.read_csv(pitstop_path))

    print(f"Wrote {len(list(FIG_DIR.glob('*.png')))} figures to {FIG_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                         default=ROOT / "data" / "processed" / "2026_master_sample.csv")
    args = parser.parse_args()
    run(args.input)
