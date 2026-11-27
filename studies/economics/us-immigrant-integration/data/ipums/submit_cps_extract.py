#!/usr/bin/env python3
"""Submit, poll, download, and verify IPUMS CPS Monthly ORG extract (2015-2026)."""

import gzip
import json
import re
import sys
from pathlib import Path

import pandas as pd
from ipumspy import IpumsApiClient, MicrodataExtract, Variable

ROOT = Path(__file__).resolve().parents[2]
API_KEY_PATH = ROOT / "data" / "ipums" / ".api_key"
OUT_DIR = ROOT / "data" / "ipums" / "raw" / "cps_org_2015_2026"

VARIABLES = [
    Variable("YEAR"),
    Variable("MONTH"),
    Variable("CPSIDP"),
    Variable("WTFINL"),
    Variable("EARNWT"),
    Variable("STATEFIP"),
    Variable("METFIPS"),
    Variable("SEX"),
    Variable("AGE", case_selections=[str(a) for a in range(18, 65)]),
    Variable("RACE"),
    Variable("HISPAN"),
    Variable("MARST"),
    Variable("EDUC"),
    Variable("BPL"),
    Variable("NATIVITY"),
    Variable("CITIZEN"),
    Variable("YRIMMIG"),
    Variable("EMPSTAT"),
    Variable("LABFORCE"),
    Variable("CLASSWKR"),
    Variable("OCC"),
    Variable("OCC2010"),
    Variable("IND"),
    Variable("UHRSWORK1"),
    Variable("UHRSWORKORG"),
    Variable("HOURWAGE"),
    Variable("HOURWAGE2"),
    Variable("EARNWEEK"),
    Variable("EARNWEEK2"),
    Variable("PAIDHOUR"),
    Variable("ELIGORG", case_selections=["1"]),
]


def get_monthly_samples_2015_plus(client: IpumsApiClient) -> list[str]:
    all_samples = []
    for page in client.get_metadata_catalog("cps", metadata_type="samples"):
        all_samples.extend(page["data"])
    monthly = [
        s["name"]
        for s in all_samples
        if re.match(r"^cps20(1[5-9]|2[0-9])_\d{2}[bs]$", s["name"])
        and "ASEC" not in s.get("description", "")
    ]
    monthly.sort()
    return monthly


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    key = API_KEY_PATH.read_text().strip()
    client = IpumsApiClient(key)

    samples = get_monthly_samples_2015_plus(client)
    print(f"Found {len(samples)} monthly CPS samples: {samples[0]} .. {samples[-1]}", flush=True)

    extract = MicrodataExtract(
        collection="cps",
        description="CPS Monthly ORG 2015-2026 (ELIGORG=1, AGE 18-64)",
        samples=samples,
        variables=VARIABLES,
        data_format="csv",
    )

    submitted = client.submit_extract(extract)
    extract_id = submitted.extract_id
    print(f"Submitted CPS extract #{extract_id}", flush=True)

    meta_path = OUT_DIR / "extract_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "collection": "cps",
                "extract_id": extract_id,
                "n_samples": len(samples),
                "first_sample": samples[0],
                "last_sample": samples[-1],
                "samples": samples,
            },
            indent=2,
        )
    )

    print(f"Waiting for extract #{extract_id} to complete...", flush=True)
    client.wait_for_extract(submitted, collection="cps", timeout=3600, inquire_delay=15)
    print(f"Extract #{extract_id} completed! Downloading to {OUT_DIR}...", flush=True)

    client.download_extract(submitted, download_dir=OUT_DIR)

    csv_files = sorted(OUT_DIR.glob("*.csv.gz")) + sorted(OUT_DIR.glob("*.csv"))
    if not csv_files:
        print("ERROR: No CSV file found after download!", file=sys.stderr)
        sys.exit(1)

    csv_path = csv_files[0]
    print(f"Downloaded CSV: {csv_path} ({csv_path.stat().st_size / (1024**2):.2f} MB)", flush=True)

    # Verify the downloaded CSV in chunks
    total_rows = 0
    min_ym = None
    max_ym = None
    columns = None
    valid_earnweek = 0
    valid_earnweek2 = 0
    valid_hourwage = 0
    valid_hourwage2 = 0
    valid_occ = 0
    valid_occ2010 = 0
    eligorg_vals = set()

    for chunk in pd.read_csv(csv_path, chunksize=250_000, low_memory=False):
        if columns is None:
            columns = list(chunk.columns)
        total_rows += len(chunk)
        ym_series = chunk["YEAR"] * 100 + chunk["MONTH"]
        c_min = int(ym_series.min())
        c_max = int(ym_series.max())
        min_ym = c_min if min_ym is None else min(min_ym, c_min)
        max_ym = c_max if max_ym is None else max(max_ym, c_max)

        if "EARNWEEK" in chunk.columns:
            valid_earnweek += int(((chunk["EARNWEEK"].notna()) & (chunk["EARNWEEK"] < 9999.99) & (chunk["EARNWEEK"] > 0)).sum())
        if "EARNWEEK2" in chunk.columns:
            valid_earnweek2 += int(((chunk["EARNWEEK2"].notna()) & (chunk["EARNWEEK2"] < 999999.99) & (chunk["EARNWEEK2"] > 0)).sum())
        if "HOURWAGE" in chunk.columns:
            valid_hourwage += int(((chunk["HOURWAGE"].notna()) & (chunk["HOURWAGE"] < 999.99) & (chunk["HOURWAGE"] > 0)).sum())
        if "HOURWAGE2" in chunk.columns:
            valid_hourwage2 += int(((chunk["HOURWAGE2"].notna()) & (chunk["HOURWAGE2"] < 999.99) & (chunk["HOURWAGE2"] > 0)).sum())
        if "OCC" in chunk.columns:
            valid_occ += int(((chunk["OCC"].notna()) & (chunk["OCC"] > 0)).sum())
        if "OCC2010" in chunk.columns:
            valid_occ2010 += int(((chunk["OCC2010"].notna()) & (chunk["OCC2010"] > 0) & (chunk["OCC2010"] < 9999)).sum())
        if "ELIGORG" in chunk.columns:
            eligorg_vals.update(chunk["ELIGORG"].dropna().unique().tolist())

    summary = {
        "csv_path": str(csv_path),
        "extract_id": extract_id,
        "total_rows": total_rows,
        "min_year_month": (min_ym // 100, min_ym % 100) if min_ym else None,
        "max_year_month": (max_ym // 100, max_ym % 100) if max_ym else None,
        "columns": columns,
        "eligorg_unique": sorted(eligorg_vals),
        "valid_earnweek": valid_earnweek,
        "valid_earnweek2": valid_earnweek2,
        "valid_hourwage": valid_hourwage,
        "valid_hourwage2": valid_hourwage2,
        "valid_occ": valid_occ,
        "valid_occ2010": valid_occ2010,
    }
    (OUT_DIR / "verification_summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== VERIFICATION SUMMARY ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
