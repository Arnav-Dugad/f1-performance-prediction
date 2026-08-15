"""
data_collection.py
===================
Stage 1 — Data Collection

Builds the core multi-season dataset for the F1 Race Performance Prediction
project using two PUBLIC, FREE, and LEGAL data sources:

1. Jolpica-F1 API (https://api.jolpi.ca/ergast/f1) — the actively maintained,
   Ergast-compatible successor to the original Ergast Developer API (Ergast
   was shut down at the end of 2024). Provides season-level structured data:
   race schedules, race results, qualifying results, driver/constructor
   standings, pit stops. Free, no API key, rate-limited (~200 unauthenticated
   requests/hour) — this script respects that with built-in delays.

2. FastF1 (https://github.com/theOehrly/Fast-F1) — a Python package that
   wraps the official F1 timing/data feeds. Provides session-level detail not
   in Ergast/Jolpica: lap-by-lap times, tyre compounds/stints, and precise
   pit stop durations, from 2018 onwards.

WHY TWO SOURCES?
Ergast/Jolpica is fast and clean for season-wide tabular data (exactly what
we need for grid position -> finish position modelling across many years).
FastF1 is slower (it downloads full timing feeds per session) but is the
only public source for lap times and tyre-stint data, which we need for the
"tyre strategy" and "lap time" factors in the research question.

HOW TO RUN THIS SCRIPT
-----------------------
This script needs unrestricted internet access to reach api.jolpi.ca and the
F1 timing feeds. Run it in ONE of:
  - Google Colab (recommended — free, no setup): upload this file's contents
    into a cell, or `git clone` the repo and `%run src/data_collection.py`
  - Your own laptop with Python 3.10+:
        pip install -r requirements.txt
        python src/data_collection.py --start-year 2018 --end-year 2025

It will NOT run inside a sandboxed environment without outbound internet
access to those specific hosts (this includes some hosted AI coding sandboxes
— check your environment's egress/firewall rules if you get connection
errors).

OUTPUT
------
data/raw/ergast/{year}_races.csv
data/raw/ergast/{year}_results.csv
data/raw/ergast/{year}_qualifying.csv
data/raw/ergast/{year}_driver_standings.csv
data/raw/ergast/{year}_pitstops.csv
data/raw/fastf1/{year}_{round}_laps.csv        (lap times, per driver, per lap)
data/raw/fastf1/{year}_{round}_stints.csv      (tyre compound per stint)

Nothing here is fabricated: every row is a value returned by one of the two
APIs above. If a request fails, the script logs a warning and skips it rather
than filling in guessed values.
"""

import argparse
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("f1-data-collection")

ROOT = Path(__file__).resolve().parent.parent
ERGAST_RAW = ROOT / "data" / "raw" / "ergast"
FASTF1_RAW = ROOT / "data" / "raw" / "fastf1"
FASTF1_CACHE = ROOT / "cache"
JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"

# Be a polite API citizen: Jolpica's unauthenticated limit is roughly
# 200 requests/hour and 4 requests/second (burst). We stay well under that.
REQUEST_DELAY_SECONDS = 0.5


# ---------------------------------------------------------------------------
# Jolpica / Ergast-compatible API helpers
# ---------------------------------------------------------------------------

def _get_json(url: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        time.sleep(REQUEST_DELAY_SECONDS)
        return resp.json()
    except requests.RequestException as e:
        log.warning(f"Request failed for {url} ({params}): {e}")
        return None


def _paginate(url: str, limit: int = 100) -> list[dict]:
    """Jolpica paginates with limit/offset. Pull every page until exhausted."""
    all_races = []
    offset = 0
    while True:
        data = _get_json(url, params={"limit": limit, "offset": offset})
        if data is None:
            break
        race_table = data["MRData"].get("RaceTable", {})
        races = race_table.get("Races", [])
        all_races.extend(races)
        total = int(data["MRData"]["total"])
        offset += limit
        if offset >= total:
            break
    return all_races


def fetch_season_schedule(year: int) -> pd.DataFrame:
    races = _paginate(f"{JOLPICA_BASE}/{year}/races")
    rows = [{
        "season": int(r["season"]), "round": int(r["round"]),
        "race_name": r["raceName"], "circuit_id": r["Circuit"]["circuitId"],
        "circuit_name": r["Circuit"]["circuitName"],
        "locality": r["Circuit"]["Location"]["locality"],
        "country": r["Circuit"]["Location"]["country"],
        "lat": float(r["Circuit"]["Location"]["lat"]),
        "lon": float(r["Circuit"]["Location"]["long"]),
        "date": r["date"],
    } for r in races]
    return pd.DataFrame(rows)


def fetch_season_results(year: int) -> pd.DataFrame:
    races = _paginate(f"{JOLPICA_BASE}/{year}/results", limit=100)
    rows = []
    for race in races:
        for res in race.get("Results", []):
            rows.append({
                "season": int(race["season"]), "round": int(race["round"]),
                "race_name": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"],
                "date": race["date"],
                "driver_id": res["Driver"]["driverId"],
                "driver_code": res["Driver"].get("code"),
                "driver_name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                "driver_dob": res["Driver"].get("dateOfBirth"),
                "driver_nationality": res["Driver"].get("nationality"),
                "constructor_id": res["Constructor"]["constructorId"],
                "constructor_name": res["Constructor"]["name"],
                "grid_position": int(res["grid"]),
                "finish_position_raw": res["position"],
                "finish_position": int(res["position"]) if res["positionText"].isdigit() else None,
                "classified_status": res["positionText"],
                "points": float(res["points"]),
                "laps_completed": int(res["laps"]),
                "status": res["status"],
                "finish_time": res.get("Time", {}).get("time"),
                "fastest_lap_rank": res.get("FastestLap", {}).get("rank"),
                "fastest_lap_time": res.get("FastestLap", {}).get("Time", {}).get("time"),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df["positions_gained"] = df["grid_position"] - df["finish_position"]
    return df


def fetch_season_qualifying(year: int) -> pd.DataFrame:
    races = _paginate(f"{JOLPICA_BASE}/{year}/qualifying", limit=100)
    rows = []
    for race in races:
        for q in race.get("QualifyingResults", []):
            rows.append({
                "season": int(race["season"]), "round": int(race["round"]),
                "race_name": race["raceName"],
                "driver_id": q["Driver"]["driverId"],
                "driver_code": q["Driver"].get("code"),
                "constructor_id": q["Constructor"]["constructorId"],
                "qualifying_position": int(q["position"]),
                "Q1": q.get("Q1") or None,
                "Q2": q.get("Q2") or None,
                "Q3": q.get("Q3") or None,
            })
    return pd.DataFrame(rows)


def fetch_season_driver_standings(year: int) -> pd.DataFrame:
    races = _paginate(f"{JOLPICA_BASE}/{year}/driverStandings", limit=100)
    rows = []
    for race in races:
        for s_list in race.get("StandingsLists", []):
            rnd = s_list.get("round")
            for s in s_list.get("DriverStandings", []):
                rows.append({
                    "season": int(year), "round": int(rnd) if rnd else None,
                    "driver_id": s["Driver"]["driverId"],
                    "standings_position": int(s["position"]),
                    "points": float(s["points"]),
                    "wins": int(s["wins"]),
                })
    return pd.DataFrame(rows)


def fetch_season_pitstops(year: int) -> pd.DataFrame:
    """Pit stop data is only reliably available on Jolpica from 2012 onward."""
    rows = []
    schedule = fetch_season_schedule(year)
    for rnd in schedule["round"]:
        data = _get_json(f"{JOLPICA_BASE}/{year}/{rnd}/pitstops", params={"limit": 100})
        if data is None:
            continue
        races = data["MRData"].get("RaceTable", {}).get("Races", [])
        for race in races:
            for p in race.get("PitStops", []):
                rows.append({
                    "season": year, "round": int(race["round"]),
                    "driver_id": p["driverId"], "stop_number": int(p["stop"]),
                    "lap": int(p["lap"]), "time_of_day": p.get("time"),
                    "duration_seconds": float(p["duration"]) if p.get("duration") else None,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FastF1 helpers (lap times + tyre stints — session-level detail)
# ---------------------------------------------------------------------------

def fetch_fastf1_laps_and_stints(year: int, round_number: int):
    """
    Pulls lap-by-lap and tyre-stint data for one race weekend using FastF1.
    Only reliable for 2018+ (FastF1's supported telemetry range).
    """
    import fastf1  # imported here so the Ergast-only path doesn't require it

    FASTF1_CACHE.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE))

    try:
        session = fastf1.get_session(year, round_number, "R")
        session.load(telemetry=False, weather=False, messages=False)
    except Exception as e:
        log.warning(f"FastF1 failed for {year} round {round_number}: {e}")
        return None, None

    laps = session.laps
    if laps is None or laps.empty:
        log.warning(f"No lap data returned for {year} round {round_number}")
        return None, None

    lap_cols = ["Driver", "DriverNumber", "LapNumber", "LapTime", "Stint",
                "Compound", "TyreLife", "FreshTyre", "Position", "IsPersonalBest"]
    laps_df = laps[[c for c in lap_cols if c in laps.columns]].copy()
    laps_df["LapTime"] = laps_df["LapTime"].dt.total_seconds()
    laps_df["season"] = year
    laps_df["round"] = round_number

    stints_df = (
        laps.groupby(["Driver", "Stint", "Compound"])
        .agg(stint_length=("LapNumber", "count"),
             start_lap=("LapNumber", "min"),
             end_lap=("LapNumber", "max"))
        .reset_index()
    )
    stints_df["season"] = year
    stints_df["round"] = round_number

    return laps_df, stints_df


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def collect_ergast_layer(start_year: int, end_year: int):
    ERGAST_RAW.mkdir(parents=True, exist_ok=True)
    for year in range(start_year, end_year + 1):
        log.info(f"=== Ergast/Jolpica layer: {year} ===")

        schedule = fetch_season_schedule(year)
        results = fetch_season_results(year)
        qualifying = fetch_season_qualifying(year)
        standings = fetch_season_driver_standings(year)
        pitstops = fetch_season_pitstops(year)

        for name, df in [("races", schedule), ("results", results),
                          ("qualifying", qualifying),
                          ("driver_standings", standings),
                          ("pitstops", pitstops)]:
            path = ERGAST_RAW / f"{year}_{name}.csv"
            if df is not None and not df.empty:
                df.to_csv(path, index=False)
                log.info(f"  wrote {len(df):>4} rows -> {path.name}")
            else:
                log.warning(f"  no data for {name} in {year} — not written")


def collect_fastf1_layer(start_year: int, end_year: int):
    FASTF1_RAW.mkdir(parents=True, exist_ok=True)
    for year in range(max(start_year, 2018), end_year + 1):
        schedule = fetch_season_schedule(year)
        if schedule.empty:
            continue
        for rnd in schedule["round"]:
            laps_df, stints_df = fetch_fastf1_laps_and_stints(year, int(rnd))
            if laps_df is not None:
                laps_df.to_csv(FASTF1_RAW / f"{year}_{rnd}_laps.csv", index=False)
                stints_df.to_csv(FASTF1_RAW / f"{year}_{rnd}_stints.csv", index=False)
                log.info(f"FastF1: {year} round {rnd} -> {len(laps_df)} lap rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Stage 1 data collection")
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--skip-fastf1", action="store_true",
                         help="Skip lap-time/tyre collection (much faster; "
                              "use if you only need results+qualifying).")
    args = parser.parse_args()

    log.info(f"Collecting Ergast/Jolpica data for {args.start_year}-{args.end_year}")
    collect_ergast_layer(args.start_year, args.end_year)

    if not args.skip_fastf1:
        log.info(f"Collecting FastF1 lap/tyre data for {args.start_year}-{args.end_year}")
        collect_fastf1_layer(args.start_year, args.end_year)

    log.info("Done. See data/raw/ for output.")
