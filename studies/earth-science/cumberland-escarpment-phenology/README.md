# Where Hopkins' 1918 Bioclimatic Law Bends: Closed-Karst Cold-Air Pooling and Plateau-Cap Phenology on the Southern Cumberland Escarpment

**Author / Director:** Daksh Tyagi  
**Provenance:** Directed by Daksh Tyagi; executed by a swarm of Gemini 4 Neon agents  
**Date:** November 2026  
**Discipline:** Earth Science (Forest Phenology, Remote Sensing & Karst Geomorphology)

## Overview

Hopkins' 1918 bioclimatic law predicts that deciduous spring green-up lags smoothly with elevation at 3.28 days per 100 meters. This study tests where that linear gradient holds and where it bends across the Southern Cumberland Plateau by combining Priority-Flood basin-closure geometry, topographically corrected Sentinel-2 and NASA HLS phenology, 25-year harmonized Landsat records, and cloud-leak-filtered ECOSTRESS thermal imagery.

### Headline Verified Findings

1. **Open slopes match Hopkins' law:** Along open escarpment slopes, half-amplitude deciduous MODIS green-up averages **3.19 days per 100 m**, closely matching Hopkins' 3.28 d/100 m rule (while earlier 15% onset thresholds lapse faster due to phenological stage definition).
2. **Closed-karst cold-air pooling compresses the growing season:** Across 11 forested closed-karst blind valleys matched with open-drainage gorges, topographic closure compresses the growing season by **9.1 days** after adjusting for pair relief imbalance; binary closure adds 54% partial variance over relief and clearly outperforms local TPI.
3. **Thermal confirmation & honest bounds:** Cloud-leak-filtered ECOSTRESS passes confirm nocturnal cooling in spring, autumn, and all seven closed basins in tile `16SED` (with wind collapse verified at Lost Cove), while worst-case swath Manski bounds span zero under cloud censoring.
4. **Plateau-cap guild lag:** Flat sandstone benches lag steep rims by **2.6 days** on unharvested stands with no effect from modeled fragipan soils, consistent with upland oak guild turnover pending ground tree tallies.

---

## Deliverables

- **Manuscript:** [`paper/main.pdf`](paper/main.pdf) · LaTeX source in [`paper/main.tex`](paper/main.tex) and [`paper/sections/`](paper/sections/) · 6 publication figures in [`paper/figures/`](paper/figures/)
- **Executed Notebook:** [`Escarpment_Hopkins_Karst_v2.ipynb`](Escarpment_Hopkins_Karst_v2.ipynb)
- **Analysis Pipelines:** Stage-3 profiling modules (`profiling/p1_karst_census.py`–`p6_alphaearth_gate.py`) and v2 consolidation & figure scripts (`v2_*.py`)
- **Adversarial Peer Review:** [`review/peer_review.md`](review/peer_review.md)

---

## Reproduction & Data Availability

Derived JSON summaries (`data/v2_*.json`, `profiling/p*.json`, `profiling/scorecard.json`) and 300-DPI figures are committed directly. Raw ECOSTRESS granules (`data/ecostress/`) and local 3DEP/NLCD GeoTIFF caches (`profiling/cache/`) are excluded from Git for repository compactness.

### Environment Setup & Re-Running Live Extractions

Set your own Google Earth Engine project ID (and NASA Earthdata credentials via `~/.netrc` for ECOSTRESS AppEEARS queries):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

export EE_PROJECT="your-gee-project-id"
python v2_p1_evt_dose_response.py
python v2_ecostress_thermal_closure.py
python v2_hopkins_climate_lal.py
python v2_signalb_audit.py
python v2_alphaearth_symmetric_dynamic.py
python v2_consolidate_summary.py
python v2_make_figures.py
```

---

## Citation

See [`CITATION.bib`](CITATION.bib).
