# Auxiliary Datasets Manifest (`data/aux/`)

All files below were downloaded and verified to load cleanly in a standard Python environment (`pandas.read_stata`, `pandas.read_csv`, `pandas.read_excel`).

---

## 1. David Dorn / Autor-Dorn (`data/aux/dorn/`)
All `.dta` files load directly via `pandas.read_stata()` (no `pyreadstat` needed):

| File | Shape | Columns | Key ID Format |
| :--- | :---: | :--- | :--- |
| `occ1990dd_task_alm.dta` | `(330, 4)` | `occ1990dd`, `task_abstract`, `task_routine`, `task_manual` | numeric `occ1990dd` (e.g. `4.0`) |
| `occ1990dd_task_offshore.dta` | `(330, 2)` | `occ1990dd`, `task_offshorability` | numeric `occ1990dd` |
| `occ1970_occ1990dd.dta` | `(443, 2)` | `occ`, `occ1990dd` | numeric `occ` |
| `occ1980_occ1990dd.dta` | `(504, 2)` | `occ`, `occ1990dd` | numeric `occ` |
| `occ1990_occ1990dd.dta` | `(502, 2)` | `occ`, `occ1990dd` | numeric `occ` |
| `occ2000_occ1990dd.dta` | `(471, 2)` | `occ`, `occ1990dd` | numeric `occ` |
| `occ2010_occ1990dd.dta` | `(489, 2)` | `occ`, `occ1990dd` | string 4-digit `occ` (e.g. `'0000'`) |

---

## 2. AI / LLM Occupational Exposure (`data/aux/ai_exposure/`)

| File | Shape | Key Columns | Key ID Format |
| :--- | :---: | :--- | :--- |
| `occ_level.csv` (Eloundou et al. 2024, OpenAI) | `(923, 8)` | `O*NET-SOC Code`, `Title`, `dv_rating_{alpha,beta,gamma}`, `human_rating_{alpha,beta,gamma}` | `11-1011.00` (strip `.00` -> 6-digit SOC for IPUMS `OCCSOC`) |
| `AIOE_DataAppendix.xlsx` (Felten et al.) | Sheet `Appendix A`: `(774, 3)`<br>Sheet `Appendix B`: `(250, 3)` | `SOC Code`, `Occupation Title`, `AIOE`<br>`NAICS`, `Industry Title`, `AIIE` | 6-digit SOC (`11-1011`) / NAICS |

---

## 3. O*NET 31.0 (`data/aux/onet/`)
Extracted from `https://www.onetcenter.org/dl_files/database/db_31_0_text.zip` (tab-delimited `.txt`):

| File | Shape | Key Columns | Key ID Format |
| :--- | :---: | :--- | :--- |
| `Occupation Data.txt` | `(1016, 3)` | `O*NET-SOC Code`, `Title`, `Description` | `11-1011.00` |
| `Work Activities.txt` | `(74702, 13)` | `O*NET-SOC Code`, `Element ID`, `Element Name`, `Scale ID`, `Data Value`, `N` | `11-1011.00` |
| `Abilities.txt` | `(94640, 13)` | `O*NET-SOC Code`, `Element ID`, `Element Name`, `Scale ID`, `Data Value`, `N` | `11-1011.00` |

---

## 4. Housing Prices & Rents (`data/aux/housing/`)

| File | Shape | Schema / Coverage | Key ID Format |
| :--- | :---: | :--- | :--- |
| `hpi_at_metro.csv` (FHFA All-Transactions) | `(84460, 6)` | Headerless: `[MSA Name, CBSA, Year, Quarter, HPI, SE]` (`1975 Q1` – `2026 Q2`) | 5-digit integer CBSA (e.g. `10180` -> merges to IPUMS `MET2013`/`MET2023`) |
| `Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv` (Zillow ZHVI) | `(895, 325)` | `RegionID`, `SizeRank`, `RegionName`, `RegionType`, `StateName`, monthly `2000-03-31` – `2026-10-31` | Metro `RegionName` / `RegionID` |
| `Metro_zori_uc_sfrcondomfr_sm_sa_month.csv` (Zillow ZORI) | `(753, 147)` | `RegionID`, `SizeRank`, `RegionName`, `RegionType`, `StateName`, monthly `2015-01-31` – `2026-10-31` | Metro `RegionName` / `RegionID` |
