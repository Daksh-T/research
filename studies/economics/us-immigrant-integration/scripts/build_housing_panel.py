#!/usr/bin/env python3
"""
Build harmonized annual MSA housing price/rent panel (FHFA HPI + Zillow ZHVI/ZORI)
keyed by IPUMS MET2013 (5-digit CBSA), and build the IPUMS METAREA <-> MET2013 <-> FHFA/Zillow
crosswalk (`data/aux/song_msa_crosswalk.csv`).

Outputs:
  - data/aux/clean_metro_housing.parquet
  - data/aux/song_msa_crosswalk.csv
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
HOUSING_DIR = ROOT / "data" / "aux" / "housing"
OUT_PARQUET = ROOT / "data" / "aux" / "clean_metro_housing.parquet"
OUT_CROSSWALK = ROOT / "data" / "aux" / "song_msa_crosswalk.csv"

# Exact OMB mapping from the 37 FHFA Metropolitan Divisions (MSADs) to parent MET2013 CBSA
MSAD_TO_PARENT_CBSA: dict[int, tuple[int, str]] = {
    # Los Angeles-Long Beach-Anaheim, CA (31080)
    11244: (31080, "Los Angeles-Long Beach-Anaheim, CA"),
    31084: (31080, "Los Angeles-Long Beach-Anaheim, CA"),
    # Washington-Arlington-Alexandria, DC-VA-MD-WV (47900)
    11694: (47900, "Washington-Arlington-Alexandria, DC-VA-MD-WV"),
    23224: (47900, "Washington-Arlington-Alexandria, DC-VA-MD-WV"),
    47764: (47900, "Washington-Arlington-Alexandria, DC-VA-MD-WV"),
    # Atlanta-Sandy Springs-Roswell, GA (12060)
    12054: (12060, "Atlanta-Sandy Springs-Roswell, GA"),
    31924: (12060, "Atlanta-Sandy Springs-Roswell, GA"),
    # Boston-Cambridge-Newton, MA-NH (14460)
    14454: (14460, "Boston-Cambridge-Newton, MA-NH"),
    15764: (14460, "Boston-Cambridge-Newton, MA-NH"),
    40484: (14460, "Boston-Cambridge-Newton, MA-NH"),
    # Philadelphia-Camden-Wilmington, PA-NJ-DE-MD (37980)
    15804: (37980, "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    33874: (37980, "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    37964: (37980, "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    48864: (37980, "Philadelphia-Camden-Wilmington, PA-NJ-DE-MD"),
    # Chicago-Naperville-Elgin, IL-IN-WI (16980)
    16984: (16980, "Chicago-Naperville-Elgin, IL-IN-WI"),
    20994: (16980, "Chicago-Naperville-Elgin, IL-IN-WI"),
    29404: (16980, "Chicago-Naperville-Elgin, IL-IN-WI"),
    29414: (16980, "Chicago-Naperville-Elgin, IL-IN-WI"),
    # Dallas-Fort Worth-Arlington, TX (19100)
    19124: (19100, "Dallas-Fort Worth-Arlington, TX"),
    23104: (19100, "Dallas-Fort Worth-Arlington, TX"),
    # Detroit-Warren-Dearborn, MI (19820)
    19804: (19820, "Detroit-Warren-Dearborn, MI"),
    47664: (19820, "Detroit-Warren-Dearborn, MI"),
    # Seattle-Tacoma-Bellevue, WA (42660)
    21794: (42660, "Seattle-Tacoma-Bellevue, WA"),
    42644: (42660, "Seattle-Tacoma-Bellevue, WA"),
    45104: (42660, "Seattle-Tacoma-Bellevue, WA"),
    # Miami-Fort Lauderdale-West Palm Beach, FL (33100)
    22744: (33100, "Miami-Fort Lauderdale-West Palm Beach, FL"),
    33124: (33100, "Miami-Fort Lauderdale-West Palm Beach, FL"),
    48424: (33100, "Miami-Fort Lauderdale-West Palm Beach, FL"),
    # New York-Newark-Jersey City, NY-NJ-PA (35620)
    29484: (35620, "New York-Newark-Jersey City, NY-NJ-PA"),
    35004: (35620, "New York-Newark-Jersey City, NY-NJ-PA"),
    35084: (35620, "New York-Newark-Jersey City, NY-NJ-PA"),
    35614: (35620, "New York-Newark-Jersey City, NY-NJ-PA"),
    # San Francisco-Oakland-Hayward, CA (41860)
    36084: (41860, "San Francisco-Oakland-Hayward, CA"),
    41884: (41860, "San Francisco-Oakland-Hayward, CA"),
    42034: (41860, "San Francisco-Oakland-Hayward, CA"),
    # Tampa-St. Petersburg-Clearwater, FL (45300)
    41304: (45300, "Tampa-St. Petersburg-Clearwater, FL"),
    45294: (45300, "Tampa-St. Petersburg-Clearwater, FL"),
}

# OMB 2023 -> 2013 CBSA code harmonization for FHFA CBSAs that changed codes
FHFA_CBSA_TO_MET2013: dict[int, int] = {
    17410: 17460,  # Cleveland, OH -> Cleveland-Elyria, OH
    19430: 19380,  # Dayton-Kettering-Beavercreek, OH -> Dayton, OH
    28880: 39100,  # Kiryas Joel-Poughkeepsie-Newburgh, NY -> Poughkeepsie-Newburgh-Middletown, NY
    30500: 15680,  # Lexington Park, MD -> California-Lexington Park, MD
    39150: 39140,  # Prescott Valley-Prescott, AZ -> Prescott, AZ
    48680: 45540,  # Wildwood-The Villages, FL -> The Villages, FL
    26180: 46520,  # Honolulu -> Urban Honolulu, HI
}


def fetch_ipums_categories(var: str) -> list[dict]:
    """Fetch official IPUMS USA variable categories JSON from usa.ipums.org."""
    url = f"https://usa.ipums.org/usa-action/variables/{var}"
    html = urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "ignore")
    m = re.search(r"categories:\s*(\[\{.*?\}\])\s*,\s*\n", html, re.S)
    if not m:
        m = re.search(r"categories:\s*(\[\{.*?\}\])", html)
    if not m:
        raise RuntimeError(f"Could not parse categories JSON for {var}")
    return json.loads(m.group(1))


def normalize_city_key(label: str) -> tuple[str, set[str]]:
    """Extract primary city token and set of state abbreviations from an MSA label."""
    if not isinstance(label, str):
        return "", set()
    # Remove parentheticals like (MSAD) or (Metropolitan Statistical Area)
    clean = re.sub(r"\([^)]*\)", "", label).strip()
    if "," in clean:
        city_part, state_part = clean.rsplit(",", 1)
        states = {s.strip().upper() for s in re.split(r"[-/\s]+", state_part) if len(s.strip()) == 2}
    else:
        city_part, states = clean, set()
    # Take first principal city before '-' or '/'
    primary_city = re.split(r"[-/]", city_part)[0].strip().lower()
    primary_city = re.sub(r"\bst\.\s*", "st ", primary_city)
    primary_city = re.sub(r"\bsaint\s+", "st ", primary_city)
    primary_city = re.sub(r"^urban\s+", "", primary_city)
    primary_city = re.sub(r"[^a-z0-9\s]", "", primary_city)
    primary_city = re.sub(r"\s+", " ", primary_city).strip()
    return primary_city, states


def all_city_tokens(label: str) -> list[str]:
    """Extract all principal city tokens (separated by '-' or '/') before the state comma."""
    if not isinstance(label, str):
        return []
    clean = re.sub(r"\([^)]*\)", "", label).strip()
    city_part = clean.rsplit(",", 1)[0] if "," in clean else clean
    tokens = []
    for tok in re.split(r"[-/]", city_part):
        t = tok.strip().lower()
        t = re.sub(r"\bst\.\s*", "st ", t)
        t = re.sub(r"\bsaint\s+", "st ", t)
        t = re.sub(r"^urban\s+", "", t)
        t = re.sub(r"[^a-z0-9\s]", "", t)
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            tokens.append(t)
    return tokens


def load_fhfa() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load FHFA HPI, harmonize MSADs & 2023->2013 CBSA codes, aggregate to annual."""
    fhfa = pd.read_csv(
        HOUSING_DIR / "hpi_at_metro.csv",
        header=None,
        names=["fhfa_msa_name", "fhfa_cbsa", "year", "quarter", "hpi", "se"],
        dtype={"fhfa_msa_name": str, "fhfa_cbsa": int, "year": int, "quarter": int},
    )
    fhfa["hpi"] = pd.to_numeric(fhfa["hpi"], errors="coerce")
    fhfa["se"] = pd.to_numeric(fhfa["se"], errors="coerce")
    fhfa = fhfa.dropna(subset=["hpi"]).copy()
    fhfa = fhfa[fhfa["hpi"] > 0].copy()

    # Map MSADs to parent CBSA
    fhfa["is_msad"] = fhfa["fhfa_cbsa"].isin(MSAD_TO_PARENT_CBSA)
    fhfa["met2013"] = fhfa["fhfa_cbsa"].map(
        lambda c: MSAD_TO_PARENT_CBSA[c][0] if c in MSAD_TO_PARENT_CBSA else FHFA_CBSA_TO_MET2013.get(c, c)
    )
    fhfa["cbsa_name"] = fhfa.apply(
        lambda r: MSAD_TO_PARENT_CBSA[r["fhfa_cbsa"]][1]
        if r["fhfa_cbsa"] in MSAD_TO_PARENT_CBSA
        else r["fhfa_msa_name"],
        axis=1,
    )

    # First average quarterly HPI within each original FHFA unit & year, then log, then average across MSADs within parent CBSA
    unit_annual = (
        fhfa.groupby(["met2013", "fhfa_cbsa", "year"], as_index=False)
        .agg(hpi_unit=("hpi", "mean"), n_quarters=("hpi", "count"))
    )
    unit_annual["ln_hpi_unit"] = np.log(unit_annual["hpi_unit"])

    annual = (
        unit_annual.groupby(["met2013", "year"], as_index=False)
        .agg(
            ln_hpi=("ln_hpi_unit", "mean"),
            hpi_n_msads=("fhfa_cbsa", "nunique"),
            hpi_min_quarters=("n_quarters", "min"),
        )
    )
    annual["hpi_annual"] = np.exp(annual["ln_hpi"])

    # Meta table of FHFA CBSAs
    meta = (
        fhfa.groupby("met2013", as_index=False)
        .agg(
            fhfa_name=("cbsa_name", "first"),
            fhfa_cbsa_codes=("fhfa_cbsa", lambda s: ";".join(str(x) for x in sorted(set(s)))),
            is_msad_parent=("is_msad", "any"),
            fhfa_first_year=("year", "min"),
            fhfa_last_year=("year", "max"),
        )
    )
    return annual, meta


def load_zillow_annual(path: Path, value_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load Zillow monthly Metro CSV and aggregate to annual means."""
    df = pd.read_csv(path)
    df = df[df["RegionType"] == "msa"].copy()
    id_cols = ["RegionID", "SizeRank", "RegionName", "StateName"]
    date_cols = [c for c in df.columns if re.match(r"^\d{4}-\d{2}-\d{2}$", c)]

    long = df.melt(id_vars=id_cols, value_vars=date_cols, var_name="date", value_name=value_col)
    long = long.dropna(subset=[value_col]).copy()
    long["year"] = long["date"].str.slice(0, 4).astype(int)
    long = long[long[value_col] > 0].copy()

    ann = (
        long.groupby(["RegionID", "year"], as_index=False)
        .agg(**{value_col: (value_col, "mean"), f"{value_col}_months": (value_col, "count")})
    )
    ann[f"ln_{value_col}"] = np.log(ann[value_col])

    meta = df[id_cols].drop_duplicates(subset=["RegionID"]).copy()
    return ann, meta


def match_zillow_to_cbsa(zillow_meta: pd.DataFrame, cbsa_meta: pd.DataFrame) -> pd.DataFrame:
    """Match Zillow metro RegionIDs to 5-digit MET2013 CBSA codes via city+state + manual overrides."""
    # Manual overrides for known naming discrepancies between Zillow and OMB/MET2013
    MANUAL_ZILLOW_TO_CBSA: dict[str, int] = {
        "New York, NY": 35620,
        "Los Angeles, CA": 31080,
        "Chicago, IL": 16980,
        "Dallas, TX": 19100,
        "Houston, TX": 26420,
        "Washington, DC": 47900,
        "Philadelphia, PA": 37980,
        "Miami, FL": 33100,
        "Atlanta, GA": 12060,
        "Boston, MA": 14460,
        "Phoenix, AZ": 38060,
        "San Francisco, CA": 41860,
        "Riverside, CA": 40140,
        "Detroit, MI": 19820,
        "Seattle, WA": 42660,
        "Minneapolis, MN": 33460,
        "San Diego, CA": 41740,
        "Tampa, FL": 45300,
        "Denver, CO": 19740,
        "St. Louis, MO": 41180,
        "Baltimore, MD": 12580,
        "Charlotte, NC": 16740,
        "Orlando, FL": 36740,
        "San Antonio, TX": 41700,
        "Portland, OR": 38900,
        "Sacramento, CA": 40900,
        "Pittsburgh, PA": 38300,
        "Las Vegas, NV": 29820,
        "Austin, TX": 12420,
        "Cincinnati, OH": 17140,
        "Kansas City, MO": 28140,
        "Columbus, OH": 18140,
        "Indianapolis, IN": 26900,
        "Cleveland, OH": 17460,
        "San Jose, CA": 41940,
        "Nashville, TN": 34980,
        "Virginia Beach, VA": 47260,
        "Providence, RI": 39300,
        "Milwaukee, WI": 33340,
        "Jacksonville, FL": 27260,
        "Oklahoma City, OK": 36420,
        "Raleigh, NC": 39580,
        "Memphis, TN": 32820,
        "Richmond, VA": 40060,
        "New Orleans, LA": 35380,
        "Louisville, KY": 31140,
        "Salt Lake City, UT": 41620,
        "Hartford, CT": 25540,
        "Buffalo, NY": 15380,
        "Birmingham, AL": 13820,
        "Rochester, NY": 40380,
        "Grand Rapids, MI": 24340,
        "Tucson, AZ": 46060,
        "Urban Honolulu, HI": 46520,
        "Honolulu, HI": 46520,
        "Tulsa, OK": 46140,
        "Fresno, CA": 23420,
        "Bridgeport, CT": 14860,
        "Stamford, CT": 14860,
        "Worcester, MA": 49340,
        "Albuquerque, NM": 10740,
        "Omaha, NE": 36540,
        "Albany, NY": 10580,
        "Bakersfield, CA": 12540,
        "New Haven, CT": 35300,
        "Knoxville, TN": 28940,
        "McAllen, TX": 32580,
        "Ventura, CA": 37100,
        "Oxnard, CA": 37100,
        "Allentown, PA": 10900,
        "El Paso, TX": 21340,
        "Baton Rouge, LA": 12940,
        "Dayton, OH": 19380,
        "Columbia, SC": 17900,
        "North Port, FL": 35840,
        "Sarasota, FL": 35840,
        "Charleston, SC": 16700,
        "Greensboro, NC": 24660,
        "Cape Coral, FL": 15980,
        "Stockton, CA": 44700,
        "Boise City, ID": 14260,
        "Colorado Springs, CO": 17820,
        "Little Rock, AR": 30780,
        "Lakeland, FL": 29460,
        "Akron, OH": 10420,
        "Des Moines, IA": 19780,
        "Springfield, MA": 44140,
        "Poughkeepsie, NY": 39100,
        "Ogden, UT": 36260,
        "Madison, WI": 31540,
        "Winston-Salem, NC": 49180,
        "Winston, NC": 49180,
        "Deltona, FL": 19660,
        "Syracuse, NY": 45060,
        "Provo, UT": 39340,
        "Toledo, OH": 45780,
        "Wichita, KS": 48620,
        "Augusta, GA": 12260,
        "Palm Bay, FL": 37340,
        "Jackson, MS": 27140,
        "Durham, NC": 20500,
        "Harrisburg, PA": 25420,
        "Spokane, WA": 44060,
        "Scranton, PA": 42540,
        "Chattanooga, TN": 16860,
        "Modesto, CA": 33700,
        "Lancaster, PA": 29540,
        "Portland, ME": 38860,
        "Youngstown, OH": 49660,
        "Lansing, MI": 29620,
        "Fayetteville, AR": 22220,
        "Lexington, KY": 30460,
        "Pensacola, FL": 37860,
        "Santa Rosa, CA": 42220,
        "Reno, NV": 39900,
        "Huntsville, AL": 26620,
        "Port St. Lucie, FL": 38940,
        "Lafayette, LA": 29180,
        "Myrtle Beach, SC": 34820,
        "Springfield, MO": 44180,
        "Visalia, CA": 47300,
        "Killeen, TX": 28660,
        "Asheville, NC": 11700,
        "York, PA": 49620,
        "Vallejo, CA": 46700,
        "Santa Maria, CA": 42200,
        "Salinas, CA": 41500,
        "Salem, OR": 41420,
        "Mobile, AL": 33660,
        "Reading, PA": 39740,
        "Corpus Christi, TX": 18580,
        "Brownsville, TX": 15180,
        "Gulfport, MS": 25060,
        "Fort Wayne, IN": 23060,
        "Savannah, GA": 42340,
        "Canton, OH": 15940,
        "Beaumont, TX": 13140,
        "Shreveport, LA": 43340,
        "Flint, MI": 22420,
        "Peoria, IL": 37900,
        "Davenport, IA": 19340,
        "Montgomery, AL": 33860,
        "Tallahassee, FL": 45220,
        "Eugene, OR": 21660,
        "Ocala, FL": 36100,
        "Hickory, NC": 25860,
        "Naples, FL": 34940,
        "Ann Arbor, MI": 11460,
        "Anchorage, AK": 11260,
        "Wilmington, NC": 48900,
        "Trenton, NJ": 45940,
        "Fort Collins, CO": 22660,
        "Rockford, IL": 40420,
        "South Bend, IN": 43780,
        "Lubbock, TX": 31180,
        "Evansville, IN": 21780,
        "Gainesville, FL": 23540,
        "Boulder, CO": 14500,
        "Lincoln, NE": 30700,
        "Green Bay, WI": 24580,
        "Spartanburg, SC": 43900,
        "Roanoke, VA": 40220,
        "Kingsport, TN": 28700,
        "Kalamazoo, MI": 28020,
        "Columbus, GA": 17980,
        "Utica, NY": 46540,
        "Duluth, MN": 20260,
        "Erie, PA": 21500,
        "San Luis Obispo, CA": 42020,
        "Santa Cruz, CA": 42100,
        "Norwich, CT": 35980,
        "Atlantic City, NJ": 12100,
        "Clarksville, TN": 17300,
        "Bremerton, WA": 14740,
        "Olympia, WA": 36500,
        "Cedar Rapids, IA": 16300,
        "Waco, TX": 47380,
        "Amarillo, TX": 11100,
        "Laredo, TX": 29700,
        "Binghamton, NY": 13780,
        "Yakima, WA": 49420,
        "Merced, CA": 32900,
        "Topeka, KS": 45820,
        "Macon, GA": 31420,
        "Sioux Falls, SD": 43620,
        "Champaign, IL": 16580,
        "Fargo, ND": 22020,
        "Bellingham, WA": 13380,
        "Charleston, WV": 16620,
        "Tuscaloosa, AL": 46220,
        "Kennewick, WA": 28420,
        "Chico, CA": 17020,
        "College Station, TX": 17780,
        "Prescott, AZ": 39140,
        "Charlottesville, VA": 16820,
        "Las Cruces, NM": 29740,
        "Greeley, CO": 24540,
        "Barnstable Town, MA": 12700,
        "Burlington, VT": 15540,
        "Elkhart, IN": 21140,
        "Yuma, AZ": 49740,
        "Medford, OR": 32780,
        "Redding, CA": 39820,
        "Saginaw, MI": 40980,
        "Tyler, TX": 46340,
        "Racine, WI": 39540,
        "Appleton, WI": 11540,
        "Bloomington, IL": 14010,
        "Bloomington, IN": 14020,
        "Springfield, IL": 44100,
        "Abilene, TX": 10180,
        "Wichita Falls, TX": 48660,
        "Pueblo, CO": 39380,
        "Billings, MT": 13740,
        "Waterloo, IA": 47940,
        "Decatur, IL": 19500,
        "Muncie, IN": 34620,
        "Terre Haute, IN": 45460,
        "Lima, OH": 30620,
        "Mansfield, OH": 31900,
        "Altoona, PA": 11020,
        "Johnstown, PA": 27780,
        "State College, PA": 44300,
        "Vineland, NJ": 47220,
        "Ocean City, NJ": 36140,
        "Dover, DE": 20100,
        "Manchester, NH": 31700,
        "Pittsfield, MA": 38340,
        "Glens Falls, NY": 24020,
        "Ithaca, NY": 27060,
        "Kingston, NY": 28740,
        "Elmira, NY": 21300,
        "Watertown, NY": 48060,
        "Kankakee, IL": 28100,
    }

    # Build lookup index from cbsa_meta: (primary_city, state) -> met2013
    primary_idx: dict[tuple[str, str], int] = {}
    any_city_idx: dict[tuple[str, str], list[int]] = {}
    for _, row in cbsa_meta.iterrows():
        cid = int(row["met2013"])
        label = row["cbsa_label"]
        pc, states = normalize_city_key(label)
        for st in states:
            primary_idx[(pc, st)] = cid
            for tok in all_city_tokens(label):
                any_city_idx.setdefault((tok, st), []).append(cid)

    matched = []
    for _, row in zillow_meta.iterrows():
        rname = row["RegionName"]
        st = str(row["StateName"]).strip().upper() if pd.notna(row["StateName"]) else ""
        cid = MANUAL_ZILLOW_TO_CBSA.get(rname)
        method = "manual" if cid is not None else None
        if cid is None:
            pc, rstates = normalize_city_key(rname)
            cand_states = list(rstates) + ([st] if st and st not in rstates else [])
            for s in cand_states:
                if (pc, s) in primary_idx:
                    cid = primary_idx[(pc, s)]
                    method = "primary_city_state"
                    break
            if cid is None:
                for s in cand_states:
                    cands = any_city_idx.get((pc, s), [])
                    if len(set(cands)) == 1:
                        cid = cands[0]
                        method = "any_city_state"
                        break
        matched.append({"RegionID": int(row["RegionID"]), "met2013": cid, "zillow_match_method": method})
    out = zillow_meta.merge(pd.DataFrame(matched), on="RegionID", how="left")
    return out


def build_metarea_to_met2013(metarea_cats: list[dict], met2013_cats: list[dict]) -> pd.DataFrame:
    """Build crosswalk from IPUMS METAREA (general 3-digit & detailed 4-digit) to MET2013 (5-digit CBSA)."""
    m13_df = pd.DataFrame(
        [
            {"met2013": int(c["code"]), "met2013_label": c["label"]}
            for c in met2013_cats
            if c["code"] != "00000"
        ]
    )

    # Index MET2013 by (primary_city, state) and (any_city, state)
    primary_idx: dict[tuple[str, str], int] = {}
    any_idx: dict[tuple[str, str], list[int]] = {}
    for _, r in m13_df.iterrows():
        cid = int(r["met2013"])
        pc, sts = normalize_city_key(r["met2013_label"])
        for st in sts:
            primary_idx[(pc, st)] = cid
            for tok in all_city_tokens(r["met2013_label"]):
                any_idx.setdefault((tok, st), []).append(cid)

    # Explicit overrides for historic METAREA names that differ from 2013 OMB CBSA names
    MANUAL_METAREA_TO_MET2013: dict[int, int] = {
        360: 31080,  # Anaheim-Santa Ana-Garden Grove, CA -> LA-Long Beach-Anaheim
        560: 12100,  # Atlantic City, NJ
        875: 35620,  # Bergen-Passaic, NJ -> NY-Newark-Jersey City
        1160: 14860,  # Bridgeport, CT -> Bridgeport-Stamford-Norwalk
        1600: 16980,  # Chicago, IL
        1920: 19100,  # Dallas-Fort Worth, TX
        2000: 19380,  # Dayton-Springfield, OH
        2080: 19740,  # Denver-Boulder, CO
        2160: 19820,  # Detroit, MI
        2680: 33100,  # Fort Lauderdale-Hollywood-Pompano Beach, FL -> Miami-Fort Lauderdale-WPB
        2800: 19100,  # Fort Worth-Arlington, TX -> Dallas-Fort Worth-Arlington
        2960: 16980,  # Gary-Hammond-East Chicago, IN -> Chicago-Naperville-Elgin
        3120: 24660,  # Greensboro-Winston Salem-High Point, NC -> Greensboro-High Point
        3280: 25540,  # Hartford-Bristol-Middleton-New Britain, CT
        3320: 46520,  # Honolulu, HI -> Urban Honolulu, HI
        3640: 35620,  # Jersey City, NJ -> NY-Newark-Jersey City
        4480: 31080,  # Los Angeles-Long Beach, CA -> LA-Long Beach-Anaheim
        4520: 31140,  # Louisville, KY/IN
        5000: 33100,  # Miami-Hialeah, FL -> Miami-Fort Lauderdale-WPB
        5015: 35620,  # Middlesex-Somerset-Hunterdon, NJ -> NY-Newark-Jersey City
        5120: 33460,  # Minneapolis-St. Paul, MN
        5190: 35620,  # Monmouth-Ocean, NJ -> NY-Newark-Jersey City
        5380: 35620,  # Nassau-Suffolk, NY -> NY-Newark-Jersey City
        5480: 35300,  # New Haven-Meriden, CT -> New Haven-Milford, CT
        5600: 35620,  # New York-Northeastern NJ -> NY-Newark-Jersey City
        5640: 35620,  # Newark, NJ -> NY-Newark-Jersey City
        5720: 47260,  # Norfolk-VA Beach-Newport News, VA -> Virginia Beach-Norfolk-Newport News
        5775: 41860,  # Oakland, CA -> San Francisco-Oakland-Hayward
        5950: 35620,  # Orange County, NY -> NY-Newark-Jersey City
        6000: 37100,  # Oxnard-Ventura, CA -> Oxnard-Thousand Oaks-Ventura
        6160: 37980,  # Philadelphia, PA/NJ
        6440: 38900,  # Portland, OR-WA
        6480: 39300,  # Providence-Fall River-Pawtucket, MA/RI
        6640: 39580,  # Raleigh-Durham, NC -> Raleigh, NC
        6780: 40140,  # Riverside-San Bernardino, CA -> Riverside-San Bernardino-Ontario
        7040: 41180,  # St. Louis, MO-IL
        7360: 41860,  # San Francisco-Oakland-Vallejo, CA -> SF-Oakland-Hayward
        7400: 41940,  # San Jose, CA -> San Jose-Sunnyvale-Santa Clara
        7480: 42200,  # Santa Barbara-Santa Maria-Lompoc, CA -> Santa Maria-Santa Barbara
        7510: 35840,  # Sarasota, FL -> North Port-Sarasota-Bradenton
        7600: 42660,  # Seattle-Everett, WA -> Seattle-Tacoma-Bellevue
        8040: 14860,  # Stamford, CT -> Bridgeport-Stamford-Norwalk
        8200: 42660,  # Tacoma, WA -> Seattle-Tacoma-Bellevue
        8280: 45300,  # Tampa-St. Petersburg-Clearwater, FL
        8720: 46700,  # Vallejo-Fairfield-Napa, CA -> Vallejo-Fairfield, CA
        8780: 47300,  # Visalia-Tulare-Porterville, CA -> Visalia-Porterville, CA
        8840: 47900,  # Washington, DC/MD/VA
        8960: 33100,  # West Palm Beach-Boca Raton-Delray Beach, FL -> Miami-Fort Lauderdale-WPB
        9160: 37980,  # Wilmington, DE/NJ/MD -> Philadelphia-Camden-Wilmington
        1120: 14460,  # Boston, MA-NH
        1125: 14500,  # Boulder-Longmont, CO
        2320: 21340,  # El Paso, TX
        3360: 26420,  # Houston-Brazoria, TX
        7160: 41620,  # Salt Lake City-Ogden, UT -> Salt Lake City, UT
        7560: 42540,  # Scranton-Wilkes-Barre, PA -> Scranton--Wilkes-Barre--Hazleton
        2655: 22500,  # Florence, SC
        3660: 27740,  # Johnson City-Kingsport-Bristol, TN/VA
        3810: 28660,  # Killeen-Temple, TX
        4880: 32580,  # McAllen-Edinburg-Pharr-Mission, TX
        6880: 40420,  # Rockford, IL
        7680: 43340,  # Shreveport, LA
        7800: 43780,  # South Bend-Mishawaka, IN
        8120: 44700,  # Stockton, CA
        8160: 45060,  # Syracuse, NY
        8400: 45780,  # Toledo, OH/MI
        8520: 46060,  # Tucson, AZ
        8560: 46140,  # Tulsa, OK
        8680: 46540,  # Utica-Rome, NY
        9040: 48620,  # Wichita, KS
        9240: 49340,  # Worcester, MA
        9320: 49660,  # Youngstown-Warren, OH-PA
        1680: 17460,  # Cleveland, OH
        1640: 17140,  # Cincinnati-Hamilton, OH/KY/IN
        1840: 18140,  # Columbus, OH
        3480: 26900,  # Indianapolis, IN
        3760: 28140,  # Kansas City, MO-KS
        5080: 33340,  # Milwaukee, WI
        5560: 35380,  # New Orleans, LA
        5880: 36420,  # Oklahoma City, OK
        5920: 36540,  # Omaha, NE/IA
        5960: 36740,  # Orlando, FL
        6200: 38060,  # Phoenix, AZ
        6280: 38300,  # Pittsburgh, PA
        6760: 40060,  # Richmond-Petersburg, VA
        6840: 40380,  # Rochester, NY
        6920: 40900,  # Sacramento, CA
        7240: 41700,  # San Antonio, TX
        7320: 41740,  # San Diego, CA
    }

    rows = []
    for c in metarea_cats:
        if not c.get("general") or c["general_code"] == "000":
            continue
        code4 = int(c["code"])
        code3 = int(c["general_code"])
        label = c["general_label"]
        cid = MANUAL_METAREA_TO_MET2013.get(code4)
        method = "manual" if cid is not None else None
        if cid is None:
            pc, sts = normalize_city_key(label)
            for st in sts:
                if (pc, st) in primary_idx:
                    cid = primary_idx[(pc, st)]
                    method = "primary_city_state"
                    break
            if cid is None:
                for st in sts:
                    for tok in all_city_tokens(label):
                        cands = any_idx.get((tok, st), [])
                        if len(set(cands)) == 1:
                            cid = cands[0]
                            method = "any_city_state"
                            break
                    if cid is not None:
                        break
        rows.append(
            {
                "metarea_gen": code3,
                "metarea_det": code4,
                "metarea_label": label,
                "met2013": cid,
                "metarea_match_method": method,
            }
        )
    df = pd.DataFrame(rows)
    df = df.merge(m13_df, on="met2013", how="left")
    return df, m13_df


def main() -> None:
    print("1. Loading FHFA HPI (harmonizing 37 MSADs -> 13 parent CBSAs)...")
    fhfa_ann, fhfa_meta = load_fhfa()
    print(f"   FHFA annual rows: {len(fhfa_ann):,}, unique MET2013 CBSAs: {fhfa_ann['met2013'].nunique()}")

    print("2. Fetching official IPUMS METAREA and MET2013 codebooks...")
    metarea_cats = fetch_ipums_categories("METAREA")
    met2013_cats = fetch_ipums_categories("MET2013")
    metarea_xwalk, m13_df = build_metarea_to_met2013(metarea_cats, met2013_cats)
    print(
        f"   METAREA general codes: {len(metarea_xwalk)}, matched to MET2013: "
        f"{metarea_xwalk['met2013'].notna().sum()} ({metarea_xwalk['met2013'].notna().mean():.1%})"
    )

    # Combine MET2013 + FHFA CBSA universe for Zillow matching
    cbsa_universe = pd.concat(
        [
            m13_df.rename(columns={"met2013_label": "cbsa_label"})[["met2013", "cbsa_label"]],
            fhfa_meta.rename(columns={"fhfa_name": "cbsa_label"})[["met2013", "cbsa_label"]],
        ],
        ignore_index=True,
    ).drop_duplicates(subset=["met2013"])

    print("3. Loading & matching Zillow ZHVI (2000-2026) and ZORI (2015-2026)...")
    zhvi_ann, zhvi_meta = load_zillow_annual(
        HOUSING_DIR / "Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv", "zhvi"
    )
    zori_ann, zori_meta = load_zillow_annual(
        HOUSING_DIR / "Metro_zori_uc_sfrcondomfr_sm_sa_month.csv", "zori"
    )

    zillow_all_meta = pd.concat([zhvi_meta, zori_meta], ignore_index=True).drop_duplicates(subset=["RegionID"])
    zillow_matched = match_zillow_to_cbsa(zillow_all_meta, cbsa_universe)
    # Keep top-ranked (largest) Zillow metro if two small Zillow entries ever collide on one CBSA
    zillow_matched_dedup = (
        zillow_matched.dropna(subset=["met2013"])
        .sort_values("SizeRank")
        .drop_duplicates(subset=["met2013"])
        .copy()
    )
    zillow_matched_dedup["met2013"] = zillow_matched_dedup["met2013"].astype(int)
    print(
        f"   Zillow metros matched to unique CBSAs: {len(zillow_matched_dedup)} "
        f"(Top-100 by SizeRank matched: {zillow_matched[zillow_matched['SizeRank'] <= 100]['met2013'].notna().sum()}/100)"
    )

    zhvi_cbsa = zhvi_ann.merge(zillow_matched_dedup[["RegionID", "met2013"]], on="RegionID", how="inner").drop(
        columns=["RegionID"]
    )
    zori_cbsa = zori_ann.merge(zillow_matched_dedup[["RegionID", "met2013"]], on="RegionID", how="inner").drop(
        columns=["RegionID"]
    )

    print("4. Merging FHFA + ZHVI + ZORI into annual CBSA panel (1975-2026)...")
    panel = fhfa_ann.merge(zhvi_cbsa, on=["met2013", "year"], how="outer").merge(
        zori_cbsa, on=["met2013", "year"], how="outer"
    )
    panel = panel.merge(cbsa_universe, on="met2013", how="left")
    panel = panel.merge(
        zillow_matched_dedup[["met2013", "RegionID", "RegionName", "SizeRank"]].rename(
            columns={"RegionID": "zillow_region_id", "RegionName": "zillow_region_name", "SizeRank": "zillow_size_rank"}
        ),
        on="met2013",
        how="left",
    )
    panel = panel.sort_values(["met2013", "year"]).reset_index(drop=True)

    # Compute exact 5-year and 10-year log changes using explicit year joins (immune to missing years)
    for lag, tag in [(5, "d5"), (10, "d10")]:
        lagged = panel[["met2013", "year", "ln_hpi", "ln_zhvi", "ln_zori"]].copy()
        lagged["year"] = lagged["year"] + lag
        lagged = lagged.rename(
            columns={
                "ln_hpi": f"_l{lag}_hpi",
                "ln_zhvi": f"_l{lag}_zhvi",
                "ln_zori": f"_l{lag}_zori",
            }
        )
        panel = panel.merge(lagged, on=["met2013", "year"], how="left")
        panel[f"{tag}_ln_hpi"] = panel["ln_hpi"] - panel[f"_l{lag}_hpi"]
        panel[f"{tag}_ln_zhvi"] = panel["ln_zhvi"] - panel[f"_l{lag}_zhvi"]
        if lag == 5:
            panel[f"{tag}_ln_zori"] = panel["ln_zori"] - panel[f"_l{lag}_zori"]
        panel = panel.drop(columns=[f"_l{lag}_hpi", f"_l{lag}_zhvi", f"_l{lag}_zori"])

    # Compute cross-sectional 2019-2024 log changes per CBSA
    p19 = panel[panel["year"] == 2019][["met2013", "ln_hpi", "ln_zhvi", "ln_zori"]].rename(
        columns={"ln_hpi": "_19_hpi", "ln_zhvi": "_19_zhvi", "ln_zori": "_19_zori"}
    )
    p24 = panel[panel["year"] == 2024][["met2013", "ln_hpi", "ln_zhvi", "ln_zori"]].rename(
        columns={"ln_hpi": "_24_hpi", "ln_zhvi": "_24_zhvi", "ln_zori": "_24_zori"}
    )
    d1924 = p19.merge(p24, on="met2013", how="outer")
    d1924["d_2019_2024_ln_hpi"] = d1924["_24_hpi"] - d1924["_19_hpi"]
    d1924["d_2019_2024_ln_zhvi"] = d1924["_24_zhvi"] - d1924["_19_zhvi"]
    d1924["d_2019_2024_ln_zori"] = d1924["_24_zori"] - d1924["_19_zori"]
    panel = panel.merge(
        d1924[["met2013", "d_2019_2024_ln_hpi", "d_2019_2024_ln_zhvi", "d_2019_2024_ln_zori"]],
        on="met2013",
        how="left",
    )

    panel.to_parquet(OUT_PARQUET, index=False)
    print(f"   Saved {OUT_PARQUET} -> shape={panel.shape}")

    print("5. Building song_msa_crosswalk.csv...")
    # Decade availability flags from panel
    decade_flags = {}
    for yr in [1980, 1990, 2000, 2010, 2019, 2024]:
        sub = panel[panel["year"] == yr]
        decade_flags[f"has_fhfa_{yr}"] = set(sub.loc[sub["ln_hpi"].notna(), "met2013"])
    zhvi_cbsas = set(panel.loc[panel["ln_zhvi"].notna(), "met2013"])
    zori_cbsas = set(panel.loc[panel["ln_zori"].notna(), "met2013"])
    fhfa_cbsas = set(fhfa_meta["met2013"])

    # Merge METAREA crosswalk with FHFA & Zillow metadata
    xwalk = metarea_xwalk.merge(
        fhfa_meta[["met2013", "fhfa_name", "fhfa_cbsa_codes", "is_msad_parent", "fhfa_first_year", "fhfa_last_year"]],
        on="met2013",
        how="left",
    ).merge(
        zillow_matched_dedup[["met2013", "RegionID", "RegionName", "SizeRank"]].rename(
            columns={"RegionID": "zillow_region_id", "RegionName": "zillow_region_name", "SizeRank": "zillow_size_rank"}
        ),
        on="met2013",
        how="left",
    )
    xwalk["in_fhfa"] = xwalk["met2013"].isin(fhfa_cbsas)
    xwalk["in_zhvi"] = xwalk["met2013"].isin(zhvi_cbsas)
    xwalk["in_zori"] = xwalk["met2013"].isin(zori_cbsas)
    for k, s in decade_flags.items():
        xwalk[k] = xwalk["met2013"].isin(s)

    xwalk.to_csv(OUT_CROSSWALK, index=False)
    print(f"   Saved {OUT_CROSSWALK} -> shape={xwalk.shape}")

    # Coverage diagnostics across IPUMS MET2013 universe (311 MSAs)
    print("\n=== COVERAGE DIAGNOSTICS (IPUMS MET2013 = 311 identifiable MSAs) ===")
    m13_set = set(m13_df["met2013"])
    for yr in [1980, 1990, 2000, 2010, 2019, 2024]:
        sub = panel[(panel["year"] == yr) & (panel["met2013"].isin(m13_set))]
        n_hpi = sub["ln_hpi"].notna().sum()
        n_zhvi = sub["ln_zhvi"].notna().sum()
        n_zori = sub["ln_zori"].notna().sum()
        print(f"  Year {yr}: FHFA HPI={n_hpi:3d}/311 ({n_hpi/311:.1%}) | ZHVI={n_zhvi:3d}/311 ({n_zhvi/311:.1%}) | ZORI={n_zori:3d}/311 ({n_zori/311:.1%})")

    # Check top 25 metros by Zillow SizeRank
    print("\n=== SANITY CHECK: TOP 15 METROS IN 2024 ===")
    top15 = panel[(panel["year"] == 2024) & (panel["zillow_size_rank"] <= 15)].sort_values("zillow_size_rank")
    cols = ["zillow_size_rank", "met2013", "zillow_region_name", "hpi_annual", "zhvi", "zori", "d_2019_2024_ln_hpi", "d_2019_2024_ln_zori"]
    print(top15[cols].to_string(index=False))

    # Print any unmatched IPUMS MET2013 in FHFA
    unmatched_m13 = m13_df[~m13_df["met2013"].isin(fhfa_cbsas)]
    print(f"\nIPUMS MET2013 codes NOT in FHFA ({len(unmatched_m13)}):")
    if len(unmatched_m13) > 0:
        print(unmatched_m13.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
