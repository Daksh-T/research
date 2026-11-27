#!/usr/bin/env python3
"""
Submit, monitor, and download 3 lean IPUMS USA microdata extracts for the
Tao Song (2017) PhD dissertation replication & extension project.

Usage:
  # Validate extract JSON payloads without an API key:
  .venv/bin/python data/ipums/submit_extracts.py --dry-run

  # Submit all 3 extracts and wait/download into data/ipums/raw/:
  IPUMS_API_KEY=... .venv/bin/python data/ipums/submit_extracts.py --submit --wait-and-download

  # Or place your key in data/ipums/.api_key (gitignored) and run:
  .venv/bin/python data/ipums/submit_extracts.py --extract essay2 --submit --wait-and-download
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

from ipumspy import IpumsApiClient, MicrodataExtract, Variable
from ipumspy.api.exceptions import BadIpumsApiRequest

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "raw"
STATE_FILE = ROOT / "submitted_extracts.json"
KEY_FILE = ROOT / ".api_key"

AGE_18_64 = [f"{a:03d}" for a in range(18, 65)]


def get_api_key() -> str:
    key = os.environ.get("IPUMS_API_KEY", "").strip()
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(
            f"IPUMS_API_KEY not found. Set IPUMS_API_KEY in env or write it to {KEY_FILE}."
        )
    return key


def make_vars(names: List[str], spouse_attached: List[str] | None = None) -> List[Variable]:
    attached_set = set(spouse_attached or [])
    seen = set()
    out: List[Variable] = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        if n in attached_set:
            out.append(Variable(n, attached_characteristics=["spouse"]))
        else:
            out.append(Variable(n))
    return out


def build_extract_essay2(include_2024: bool = True) -> MicrodataExtract:
    """
    Essay 2 (1980-2024): Foreign-born adults ages 18-64 (all MARST) + attached spouse vars.
    Note: CITIZEN in [1..5] excludes Puerto Rico (BPL=110) and US territories (BPL 100-120,
    coded CITIZEN=0) at extract time to keep extract size lean (~87% row reduction).
    """
    samples = [
        "us1980a", "us1990a", "us2000a",
        "us2005a", "us2006a", "us2007a", "us2008a", "us2009a", "us2010a",
        "us2012a", "us2015a", "us2017a", "us2019a",
        "us2021a", "us2022a", "us2023a",
    ]
    if include_2024:
        samples.append("us2024a")

    # Note: IPUMS USA API automatically emits detailed BPLD/EDUCD when BPL/EDUC are requested.
    spouse_vars = [
        "BPL", "RACE", "HISPAN", "ANCESTR1",
        "EDUC", "SPEAKENG", "LANGUAGE",
        "EMPSTAT", "UHRSWORK", "INCWAGE",
        "OCC1990", "OCC2010", "CITIZEN", "AGE",
    ]
    base_vars = [
        "YEAR", "SAMPLE", "SERIAL", "PERNUM", "PERWT", "HHWT",
        "STATEFIP", "METAREA", "MET2013", "METRO", "PUMA",
        "SEX", "AGE", "MARST", "RACE", "HISPAN", "ANCESTR1",
        "BPL", "CITIZEN", "YRIMMIG", "YRSUSA1",
        "SPEAKENG", "LANGUAGE",
        "SCHOOL", "EDUC", "DEGFIELD",
        "EMPSTAT", "LABFORCE", "CLASSWKR", "UHRSWORK", "WKSWORK1", "WKSWORK2",
        "INCWAGE", "CPI99", "OCC1990", "OCC2010", "OCCSOC", "IND1990",
        "NCHILD", "RELATE", "SPLOC",
    ]
    ex = MicrodataExtract(
        collection="usa",
        description="Song PhD Eval - Essay 2 (1980-2024 foreign-born excl PR/territories, 18-64, all MARST + spouse vars)",
        samples=samples,
        variables=make_vars(base_vars, spouse_attached=spouse_vars),
        data_format="csv",
    )
    ex.select_cases("AGE", AGE_18_64, general=True)
    ex.select_cases("CITIZEN", ["1", "2", "3", "4", "5"], general=True)
    return ex


def build_extract_essay1_1970_2000() -> MicrodataExtract:
    """Historical benchmark (1970-2000) for Essays 1 & 3, ages 18-64."""
    samples = ["us1970c", "us1970d", "us1980a", "us1990a", "us2000a"]
    vars_list = [
        "YEAR", "SAMPLE", "SERIAL", "PERNUM", "PERWT", "HHWT",
        "STATEFIP", "METAREA", "MET2013", "METRO", "CONSPUMA", "PUMA",
        "CNTYGP97", "CNTYGP98",
        "SEX", "AGE", "RACE", "HISPAN", "MARST", "SCHOOL", "EDUC",
        "BPL", "CITIZEN", "YRIMMIG", "YRSUSA1", "SPEAKENG", "LANGUAGE",
        "ANCESTR1", "ANCESTR2",
        "EMPSTAT", "LABFORCE", "CLASSWKR", "UHRSWORK", "WKSWORK1", "WKSWORK2",
        "INCWAGE", "CPI99", "OCC1990", "OCC2010", "IND1990",
        "MIGRATE5", "MIGMETAREA5", "MIGPLAC5",
        "OWNERSHP", "RENTGRS", "VALUEH", "ROOMS", "BEDROOMS", "BUILTYR2",
        "UNITSSTR", "NUMPREC", "MULTGEN",
    ]
    ex = MicrodataExtract(
        collection="usa",
        description="Song PhD Eval - Essays 1 & 3 Historical (1970-2000, ages 18-64)",
        samples=samples,
        variables=make_vars(vars_list),
        data_format="csv",
    )
    ex.select_cases("AGE", AGE_18_64, general=True)
    return ex


def build_extract_acs_2007_2024(include_2024: bool = True) -> MicrodataExtract:
    """Modern ACS extension (2007-2024) for Essays 1 & 3, ages 18-64."""
    samples = [
        "us2007a", "us2010a", "us2015a", "us2018a", "us2019a",
        "us2021a", "us2022a", "us2023a",
    ]
    if include_2024:
        samples.append("us2024a")

    vars_list = [
        "YEAR", "SAMPLE", "SERIAL", "PERNUM", "PERWT", "HHWT",
        "STATEFIP", "MET2013", "CPUMA0010", "PUMA", "METRO",
        "SEX", "AGE", "MARST", "SCHOOL", "EDUC", "DEGFIELD",
        "RACE", "HISPAN", "ANCESTR1", "ANCESTR2",
        "BPL", "CITIZEN", "YRIMMIG", "YRSUSA1", "SPEAKENG", "LANGUAGE",
        "EMPSTAT", "LABFORCE", "CLASSWKR", "UHRSWORK", "WKSWORK2",
        "INCWAGE", "CPI99", "OCC1990", "OCC2010", "OCCSOC", "IND1990", "INDNAICS",
        "TRANWORK",
        "MIGRATE1", "MIGMET131", "MIGPLAC1", "MIGPUMA1",
        "OWNERSHP", "RENTGRS", "VALUEH", "ROOMS", "BEDROOMS", "BUILTYR2",
        "UNITSSTR", "NUMPREC", "MULTGEN",
    ]
    ex = MicrodataExtract(
        collection="usa",
        description="Song PhD Eval - Essays 1 & 3 Modern ACS (2007-2024, ages 18-64)",
        samples=samples,
        variables=make_vars(vars_list),
        data_format="csv",
    )
    ex.select_cases("AGE", AGE_18_64, general=True)
    return ex


BUILDERS = {
    "essay2": build_extract_essay2,
    "essay1_1970_2000": build_extract_essay1_1970_2000,
    "acs_2007_2024": build_extract_acs_2007_2024,
}


def submit_with_fallback(client: IpumsApiClient, name: str) -> MicrodataExtract:
    builder = BUILDERS[name]
    try:
        ex = builder()
        return client.submit_extract(ex)
    except BadIpumsApiRequest as e:
        if "us2024a" in str(e) and name in ("essay2", "acs_2007_2024"):
            print(f"[WARN] {name}: us2024a rejected by API, retrying without us2024a...", file=sys.stderr)
            ex = builder(include_2024=False)
            return client.submit_extract(ex)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit and download IPUMS USA extracts.")
    parser.add_argument(
        "--extract",
        choices=["all", *BUILDERS.keys()],
        default="all",
        help="Which extract(s) to operate on.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and print payload summary only.")
    parser.add_argument("--submit", action="store_true", help="Submit extract(s) to IPUMS API.")
    parser.add_argument("--wait-and-download", action="store_true", help="Wait for completion and download.")
    args = parser.parse_args()

    targets = list(BUILDERS.keys()) if args.extract == "all" else [args.extract]

    if args.dry_run:
        for name in targets:
            payload = BUILDERS[name]().build()
            print(
                f"[DRY-RUN] {name}: samples={len(payload['samples'])}, "
                f"vars={len(payload['variables'])}, format={payload['dataFormat']}"
            )
        return

    if not (args.submit or args.wait_and_download):
        parser.print_help()
        return

    client = IpumsApiClient(get_api_key())
    state: Dict[str, int] = json.loads(STATE_FILE.read_text()) if STATE_FILE.exists() else {}

    if args.submit:
        for name in targets:
            ex = submit_with_fallback(client, name)
            state[name] = ex.extract_id
            print(f"[SUBMITTED] {name} -> extract_id={ex.extract_id}")
        STATE_FILE.write_text(json.dumps(state, indent=2) + "\n")

    if args.wait_and_download:
        for name in targets:
            if name not in state:
                raise RuntimeError(f"No extract_id recorded for {name} in {STATE_FILE}. Run with --submit first.")
            eid = state[name]
            out_dir = RAW_DIR / name
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"[WAITING] {name} (extract_id={eid})...")
            client.wait_for_extract(eid, collection="usa")
            client.download_extract(eid, download_dir=out_dir, collection="usa")
            print(f"[DOWNLOADED] {name} -> {out_dir}")


if __name__ == "__main__":
    main()
