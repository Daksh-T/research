# Seeing Beneath the Canopy: Multi-Epoch Airborne LiDAR Evaluates Leaf-Off Landsat Proxies and Vertical Prescribed-Fire Effects on the Cumberland Plateau

**Author / Director:** Daksh Tyagi  
**Provenance:** Directed by Daksh Tyagi; executed by a swarm of Gemini 4 Neon agents  
**Date:** November 2026  
**Discipline:** Earth Science (Remote Sensing, Airborne LiDAR & Forest Fire Ecology)

A standalone remote-sensing and forest-structure study spanning 480 km across the Cumberland Plateau (Alabama, Tennessee, and Kentucky), combining a 40-year Landsat Collection 2 radiometric audit with 689 pure-deciduous USGS 3DEP airborne LiDAR plots and 1,110 official prescribed-fire perimeters.

---

## Core Deliverables

- **Manuscript (Typst + PDF):**
  - [`paper/cumberland_plateau_lidar_fire.pdf`](paper/cumberland_plateau_lidar_fire.pdf) — 14-page typeset paper (source in [`paper/main.typ`](paper/main.typ) and [`paper/sections/`](paper/sections/)).
  - [`paper/figures/`](paper/figures/) — 6 publication-grade vector PDF & 300-DPI PNG figures (`fig0_study_area_map`, `fig_mechanism_schematic`, `fig1`–`fig4`).
- **Interactive Web Explorer:**
  - [`explorer.html`](explorer.html) — Self-contained interactive Leaflet workbench using keyless USGS Topo, USGS Shaded Relief, Esri World Topo, Esri Satellite Imagery, and an embedded offline-ready Google Earth Engine SRTM + Annual NLCD forest-relief overlay, paired with a live 4-stratum Vertical Canopy Inspector, Evidence & Sensor Advisor, tabbed figure gallery, and sortable Tennessee protected-area table.
- **Reproducible Research Notebooks (executed with saved outputs):**
  - [`notebooks/01_optical_audit_and_region.ipynb`](notebooks/01_optical_audit_and_region.ipynb) — Landsat Collection 2 overlap harmonization audit and regional trends.
  - [`notebooks/02_lidar_fire_structure.ipynb`](notebooks/02_lidar_fire_structure.ipynb) — 689-plot 3DEP airborne LiDAR × USFS FACTS / Sewanee LAL BACI analysis, continuous 1 m vertical profiles, and optical/GEDI validation.
- **Adversarial Peer Review & Revision Log:**
  - [`review/peer_review.md`](review/peer_review.md) and [`review/response_to_referee.md`](review/response_to_referee.md).

---

## Summary of Verified Findings

1. **Neither winter nor spring Landsat NDVI tracks temporal deciduous midstory change:** Empirical overlap harmonization cuts the apparent 40-year plateau winter NDVI trend by **79%** (`+0.033` to `+0.007 / decade`), with 7 of 8 Tennessee protected areas sloping slightly negative in 2014–2025. Against repeat 3DEP LiDAR BACI plots, both winter (`r = -0.00, p = 0.98, n = 272`) and spring (`r = -0.05, p = 0.37, n = 283`) Landsat NDVI deltas fail to track 3D midstory change.
2. **Identified multi-epoch LiDAR BACI fire contrasts:** Because newer 3DEP flights record more multiple returns per pulse (`frac_first_returns`: `0.43 -> 0.32` in Bankhead; `0.59 -> 0.44` in Daniel Boone), only burned-minus-unburned DiD is identified. In Bankhead NF (2012–2021, `n = 139`), `1` and `2+` inter-epoch burns produced large, density-invariant DiD reductions (`-0.066` and `-0.076` NRD; `d = -0.82` and `-0.95`; burn-unit `p_SUID <= 0.0001`). In Daniel Boone NF (2017–2023, `n = 144`), a single burn had null DiD (`+0.009, p_SUID = 0.75`).
3. **Continuous 1 m vertical profile & unbiased first-return overstory (`n = 689`, `G = 359` burn units):** Under `3+` burns, midstory reductions attenuate smoothly with height (significant through `5–6 m`, `p_SUID = 0.020`; null `>6 m`). Using unbiased first-return canopy cover (`>10 m`), `1–3+` burns leave the overstory statistically unchanged (`p_SUID = 0.12–0.36`), while `mech+fire` opens it by `-0.067` (`p_SUID = 0.001`).

---

## Pipeline Reproduction & Data Availability

Derived plot-level LiDAR metrics (`data/lidar/`), protected-area boundaries (`data/areas.geojson`), and summary JSONs are included directly. Raw USFS FACTS prescribed-fire perimeters (`data/facts/`, ~17 MB) are excluded from Git and can be downloaded from the official [USFS EDW FACTS Hazardous Fuels Treatment Reduction dataset](https://data.fs.usda.gov/geodata/edw/datasets.php).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

export EE_PROJECT="your-gee-project-id"
python build_areas.py
python scripts/01_diagnostic_and_regional.py
python scripts/02_sample_plots.py
python scripts/03_extract_lidar_and_optical.py
python scripts/04_analyze_results.py
python scripts/05_make_figures.py
python scripts/06_build_practitioner_explorer.py
python scripts/07_make_paper_extra_figures.py
```

---

## Citation

See [`CITATION.bib`](CITATION.bib).
