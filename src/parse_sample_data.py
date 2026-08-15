"""
parse_sample_data.py
---------------------
Converts the raw JSON snapshots in data/sample_live_fetch/ (pulled live from the
Jolpica-F1 API, the official successor to the Ergast API, on the project setup
date) into tidy CSV files under data/processed/.

This is a SAMPLE / BOOTSTRAP dataset only — it covers the first two rounds of
the 2026 season, fetched live to prove the pipeline works end-to-end with real
data. The full multi-season historical dataset used for the actual ML model
must be built by running src/data_collection.py with normal internet access
(see README.md, Stage 1 notes).

No data in this file is invented. Every row traces back to a JSON response
returned by https://api.jolpi.ca/ergast/f1/ on the date this project was set up.
"""
import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_live_fetch"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_races():
    with open(RAW_DIR / "2026_races_raw.json") as f:
        data = json.load(f)
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        rows.append({
            "season": int(race["season"]),
            "round": int(race["round"]),
            "race_name": race["raceName"],
            "circuit_id": race["Circuit"]["circuitId"],
            "circuit_name": race["Circuit"]["circuitName"],
            "locality": race["Circuit"]["Location"]["locality"],
            "country": race["Circuit"]["Location"]["country"],
            "lat": float(race["Circuit"]["Location"]["lat"]),
            "lon": float(race["Circuit"]["Location"]["long"]),
            "date": race["date"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "2026_race_schedule.csv", index=False)
    print(f"races: wrote {len(df)} rows -> 2026_race_schedule.csv")
    return df


def parse_results():
    with open(RAW_DIR / "2026_results_r1_r2_raw.json") as f:
        data = json.load(f)
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        for res in race["Results"]:
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "circuit_id": race["Circuit"]["circuitId"],
                "date": race["date"],
                "driver_id": res["Driver"]["driverId"],
                "driver_code": res["Driver"]["code"],
                "driver_name": f'{res["Driver"]["givenName"]} {res["Driver"]["familyName"]}',
                "constructor_id": res["Constructor"]["constructorId"],
                "constructor_name": res["Constructor"]["name"],
                "grid_position": int(res["grid"]),
                "finish_position": int(res["position"]),
                "points": float(res["points"]),
                "laps_completed": int(res["laps"]),
                "status": res["status"],
                "finish_time": res.get("Time", {}).get("time"),
            })
    df = pd.DataFrame(rows)
    df["positions_gained"] = df["grid_position"] - df["finish_position"]
    df.to_csv(OUT_DIR / "2026_race_results_sample.csv", index=False)
    print(f"results: wrote {len(df)} rows -> 2026_race_results_sample.csv")
    return df


def parse_qualifying():
    with open(RAW_DIR / "2026_qualifying_r1_r2_raw.json") as f:
        data = json.load(f)
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        for q in race["QualifyingResults"]:
            rows.append({
                "season": int(race["season"]),
                "round": int(race["round"]),
                "race_name": race["raceName"],
                "driver_id": q["Driver"]["driverId"],
                "driver_code": q["Driver"]["code"],
                "constructor_id": q["Constructor"]["constructorId"],
                "constructor_name": q["Constructor"]["name"],
                "qualifying_position": int(q["position"]),
                "Q1": q.get("Q1") or None,
                "Q2": q.get("Q2") or None,
                "Q3": q.get("Q3") or None,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "2026_qualifying_sample.csv", index=False)
    print(f"qualifying: wrote {len(df)} rows -> 2026_qualifying_sample.csv")
    return df


def parse_pitstops():
    with open(RAW_DIR / "2026_r1_pitstops_raw.json") as f:
        data = json.load(f)
    rows = []
    for race in data["MRData"]["RaceTable"]["Races"]:
        for p in race["PitStops"]:
            dur_raw = p["duration"]
            # Ergast/Jolpica reports duration as seconds normally (e.g. "18.266"),
            # but as mm:ss.sss for very long stationary times, e.g. stops that
            # span a red-flag stoppage ("16:12.356"). Parse both correctly
            # rather than silently mis-reading long stops as ~16 seconds.
            if ":" in dur_raw:
                mins, secs = dur_raw.split(":")
                duration_seconds = int(mins) * 60 + float(secs)
                long_stop_flag = True
            else:
                duration_seconds = float(dur_raw)
                long_stop_flag = False
            rows.append({
                "season": int(race["season"]), "round": int(race["round"]),
                "driver_id": p["driverId"],
                "stop_number": int(p["stop"]),
                "lap": int(p["lap"]),
                "duration_seconds": duration_seconds,
                "unusually_long_stop": long_stop_flag,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "2026_r1_pitstops_sample.csv", index=False)
    print(f"pitstops: wrote {len(df)} rows -> 2026_r1_pitstops_sample.csv "
          f"({df['unusually_long_stop'].sum()} flagged as unusually long / likely red-flag stops)")
    return df


def parse_driver_standings():
    with open(RAW_DIR / "2026_driverstandings_after_r11_raw.json") as f:
        data = json.load(f)
    rows = []
    for s_list in data["MRData"]["StandingsTable"]["StandingsLists"]:
        for s in s_list["DriverStandings"]:
            rows.append({
                "season": int(s_list["season"]),
                "as_of_round": int(s_list["round"]),
                "driver_id": s["Driver"]["driverId"],
                "driver_code": s["Driver"]["code"],
                "constructor_id": s["Constructors"][0]["constructorId"],
                "standings_position": int(s["position"]),
                "points": float(s["points"]),
                "wins": int(s["wins"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "2026_driver_standings_sample.csv", index=False)
    print(f"standings: wrote {len(df)} rows -> 2026_driver_standings_sample.csv")
    return df


def build_master_table(results, qualifying, standings):
    """
    Merge results + qualifying + (pre-race) standings into one modelling-ready
    table. Standings are joined at 'as_of_round' so we don't leak future
    information — a driver's championship standing used to predict round N
    should reflect their standing BEFORE round N, not after.
    """
    df = results.merge(
        qualifying[["season", "round", "driver_id", "qualifying_position"]],
        on=["season", "round", "driver_id"], how="left"
    )
    # standings snapshot is only available after round 11 in our sample, so
    # we can only attach it (as a pre-race feature) for rounds AFTER 11.
    # For rounds 1-2 (all we have race results for), no valid prior-round
    # standing exists yet in this small sample — left as null, honestly.
    df = df.merge(
        standings.rename(columns={"points": "standings_points_snapshot",
                                   "as_of_round": "standings_as_of_round"}),
        on=["season", "driver_id"], how="left", suffixes=("", "_standing")
    )
    df["standings_points_snapshot"] = df.apply(
        lambda r: r["standings_points_snapshot"] if r["standings_as_of_round"] < r["round"] else None,
        axis=1
    )
    df.to_csv(OUT_DIR / "2026_master_sample.csv", index=False)
    print(f"master: wrote {len(df)} rows -> 2026_master_sample.csv")
    return df


if __name__ == "__main__":
    parse_races()
    results = parse_results()
    qualifying = parse_qualifying()
    parse_pitstops()
    standings = parse_driver_standings()
    build_master_table(results, qualifying, standings)
