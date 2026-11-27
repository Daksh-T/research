#!/usr/bin/env python3
"""Fetch, filter, harmonize, and verify Census Bureau Basic Monthly CPS ORG (2020-2026)."""

import concurrent.futures
import io
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "ipums" / "raw" / "cps_org_2015_2026"
PARQUET_PATH = OUT_DIR / "cps_org_clean.parquet"

KEEP_COLS = [
    "HRYEAR4",
    "HRMONTH",
    "HRMIS",
    "HRHHID",
    "HRHHID2",
    "PULINENO",
    "PRTAGE",
    "PESEX",
    "PEEDUCA",
    "PENATVTY",
    "PRCITSHP",
    "PTDTRACE",
    "PEHSPNON",
    "PEMARITL",
    "GESTFIPS",
    "GTCBSA",
    "PEMLR",
    "PEIO1COW",
    "PTIO1OCD",
    "PEIO1ICD",
    "PEHRUSL1",
    "PTERNWA",
    "PTERNHLY",
    "PEERNHRY",
    "PWORWGT",
    "PWCMPWGT",
    "PWSSWGT",
]
KEEP_SET = set(KEEP_COLS)

MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def discover_csv_urls(start_year: int = 2020, end_year: int = 2026) -> list[tuple[int, int, str]]:
    tasks = []
    for yr in range(start_year, end_year + 1):
        base_url = f"https://www2.census.gov/programs-surveys/cps/datasets/{yr}/basic/"
        r = requests.get(base_url, timeout=20)
        r.raise_for_status()
        csvs = sorted(set(re.findall(r'href="([a-z]{3}\d{2}pub\.csv)"', r.text, re.I)))
        for fname in csvs:
            mon_str = fname[:3].lower()
            if mon_str in MONTH_MAP:
                tasks.append((yr, MONTH_MAP[mon_str], base_url + fname))
    tasks.sort()
    return tasks


def fetch_one_month(task: tuple[int, int, str]) -> tuple[int, int, pd.DataFrame]:
    yr, mon, url = task
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
            df = pd.read_csv(
                io.BytesIO(r.content),
                usecols=lambda c: c.strip().upper() in KEEP_SET,
                low_memory=False,
            )
            df.columns = [c.strip().upper() for c in df.columns]
            # Filter to Outgoing Rotation Group (PWORWGT > 0) and ages 18-64
            for num_col in ["PWORWGT", "PRTAGE", "PTERNWA", "PTERNHLY", "PWSSWGT", "PWCMPWGT", "PTIO1OCD", "PRCITSHP", "PESEX", "PEEDUCA"]:
                if num_col in df.columns:
                    df[num_col] = pd.to_numeric(df[num_col], errors="coerce")
            org = df[(df["PWORWGT"] > 0) & (df["PRTAGE"].between(18, 64))].copy()
            # Ensure YEAR and MONTH are populated cleanly
            org["YEAR"] = yr
            org["MONTH"] = mon
            # Keep string IDs clean
            if "HRHHID" in org.columns:
                org["HRHHID"] = org["HRHHID"].astype(str)
            if "HRHHID2" in org.columns:
                org["HRHHID2"] = org["HRHHID2"].astype(str)
            return yr, mon, org
        except Exception as e:
            if attempt == 3:
                raise RuntimeError(f"Failed {yr}-{mon:02d} ({url}): {e}") from e
            time.sleep(2 ** (attempt + 1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tasks = discover_csv_urls(2020, 2026)
    print(f"Discovered {len(tasks)} monthly CPS CSVs ({tasks[0][0]}-{tasks[0][1]:02d} .. {tasks[-1][0]}-{tasks[-1][1]:02d})", flush=True)

    frames = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(fetch_one_month, t) for t in tasks]
        for fut in concurrent.futures.as_completed(futs):
            yr, mon, df = fut.result()
            frames.append(df)
            done += 1
            if done % 10 == 0 or done == len(tasks):
                print(f"  Completed {done}/{len(tasks)} months (elapsed {time.time()-t0:.1f}s)", flush=True)

    full = pd.concat(frames, ignore_index=True)
    full.sort_values(["YEAR", "MONTH"], inplace=True, ignore_index=True)

    # Harmonized derived columns
    full["EARNWEEK"] = np.where(full["PTERNWA"] > 0, full["PTERNWA"] / 100.0, np.nan)
    full["HOURWAGE"] = np.where(full["PTERNHLY"] > 0, full["PTERNHLY"] / 100.0, np.nan)
    full["EARNWT"] = full["PWORWGT"] / 10000.0
    full["WTFINL"] = full["PWSSWGT"] / 10000.0
    full["OCC_2018"] = full["PTIO1OCD"].astype("Int64")
    full["FOREIGN_BORN"] = full["PRCITSHP"].isin([4, 5]).astype("int8")
    full["FEMALE"] = (full["PESEX"] == 2).astype("int8")
    full["BA_PLUS"] = (full["PEEDUCA"] >= 43).astype("int8")

    full.to_parquet(PARQUET_PATH, index=False, compression="snappy")
    mb = PARQUET_PATH.stat().st_size / (1024 ** 2)
    print(f"\nSaved {PARQUET_PATH} ({mb:.2f} MB) in {time.time()-t0:.1f}s", flush=True)

    # Verification summary
    summary = {
        "parquet_path": str(PARQUET_PATH),
        "size_mb": round(mb, 2),
        "total_rows": int(len(full)),
        "n_months": int(full.groupby(["YEAR", "MONTH"]).ngroups),
        "min_year_month": [int(full["YEAR"].min()), int(full.loc[full["YEAR"] == full["YEAR"].min(), "MONTH"].min())],
        "max_year_month": [int(full["YEAR"].max()), int(full.loc[full["YEAR"] == full["YEAR"].max(), "MONTH"].max())],
        "columns": list(full.columns),
        "valid_earnweek_n": int(full["EARNWEEK"].notna().sum()),
        "valid_hourwage_n": int(full["HOURWAGE"].notna().sum()),
        "valid_occ2018_n": int((full["OCC_2018"] > 0).sum()),
        "earnweek_quantiles": {
            k: round(float(v), 2)
            for k, v in full["EARNWEEK"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).items()
        },
        "hourwage_quantiles": {
            k: round(float(v), 2)
            for k, v in full["HOURWAGE"].quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.99]).items()
        },
        "rows_by_year": {int(k): int(v) for k, v in full.groupby("YEAR").size().items()},
    }
    (OUT_DIR / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== VERIFICATION SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
