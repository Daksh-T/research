# Manuscript & Study Catalog

Complete index of all 5 papers across 3 studies (2 disciplines) in `Daksh-T/research`.

---

## Earth Science

### 1. Where Hopkins' 1918 Bioclimatic Law Bends: Closed-Karst Cold-Air Pooling and Plateau-Cap Phenology on the Southern Cumberland Escarpment
- **Study Directory:** [`studies/earth-science/cumberland-escarpment-phenology/`](studies/earth-science/cumberland-escarpment-phenology/)
- **Date:** November 2026
- **Manuscript:** [PDF](studies/earth-science/cumberland-escarpment-phenology/paper/main.pdf) · [LaTeX Source](studies/earth-science/cumberland-escarpment-phenology/paper/main.tex)
- **Interactive / Reproducible:** [Executed Notebook](studies/earth-science/cumberland-escarpment-phenology/Escarpment_Hopkins_Karst_v2.ipynb)
- **Adversarial Review:** [`review/peer_review.md`](studies/earth-science/cumberland-escarpment-phenology/review/peer_review.md)
- **Data Policy:** Includes derived JSON summaries and 300-DPI figures; raw thermal satellite imagery (ECOSTRESS) and elevation/land-cover layers (3DEP/NLCD) are fetched via NASA AppEEARS and Google Earth Engine (`EE_PROJECT`).
- **Summary:** A century-old rule of thumb (Hopkins' 1918 bioclimatic law) predicts that spring green-up arrives about 3.3 days later for every 100 meters of elevation gain. Across 25 years of satellite vegetation and thermal imagery on the Southern Cumberland Plateau, open hillsides closely follow that rule (3.2 days per 100 m), while two landforms bend it: 11 enclosed karst valleys trap cold air at night and shorten the growing season by 9.1 days, and flat sandstone plateau tops leaf out 2.6 days later than nearby steep rims.

### 2. Seeing Beneath the Canopy: Multi-Epoch Airborne LiDAR Evaluates Leaf-Off Landsat Proxies and Vertical Prescribed-Fire Effects on the Cumberland Plateau
- **Study Directory:** [`studies/earth-science/cumberland-plateau-lidar-fire/`](studies/earth-science/cumberland-plateau-lidar-fire/)
- **Date:** November 2026
- **Manuscript:** [PDF](studies/earth-science/cumberland-plateau-lidar-fire/paper/cumberland_plateau_lidar_fire.pdf) · [Typst Source](studies/earth-science/cumberland-plateau-lidar-fire/paper/main.typ)
- **Interactive / Reproducible:** [Interactive Leaflet Explorer](studies/earth-science/cumberland-plateau-lidar-fire/explorer.html) · [Notebooks](studies/earth-science/cumberland-plateau-lidar-fire/notebooks/)
- **Adversarial Review:** [`review/peer_review.md`](studies/earth-science/cumberland-plateau-lidar-fire/review/peer_review.md) · [`review/response_to_referee.md`](studies/earth-science/cumberland-plateau-lidar-fire/review/response_to_referee.md)
- **Data Policy:** Includes 689-plot LiDAR metrics, study boundaries, and analysis summaries; raw USFS fire perimeters (FACTS) and USGS 3DEP LiDAR point clouds are fetched via public URLs and Google Earth Engine (`EE_PROJECT`).
- **Summary:** Decades of fire suppression across eastern U.S. oak–hickory forests have let shade-tolerant trees like red maple and beech crowd the understory beneath mature oaks. Pairing 40 years of Landsat imagery with repeat airborne LiDAR scans across 689 deciduous plots and 1,110 prescribed-fire perimeters on the Cumberland Plateau, this study tests whether winter satellite greenness can track that understory crowding. It cannot: once sensor drift is accounted for, winter greenness shows virtually zero correlation with actual 3D understory change. Repeat airborne LiDAR, however, clearly detects large midstory reductions under repeated prescribed burns without opening the upper tree canopy.

---

## Economics

### Family: Three Empirical Studies on Immigrant Socio-Economic Integration in the United States
- **Study Family Directory:** [`studies/economics/us-immigrant-integration/`](studies/economics/us-immigrant-integration/)
- **Date:** November 2026
- **Adversarial Review:** [`review/peer_review.md`](studies/economics/us-immigrant-integration/review/peer_review.md) · [`review/response_to_referee.md`](studies/economics/us-immigrant-integration/review/response_to_referee.md)
- **Data Policy:** Includes summary regression outputs, LaTeX tables, figures, and crosswalks. Raw and derived person-level IPUMS microdata are excluded per IPUMS Terms of Use; reproduction scripts (`data/ipums/submit_extracts.py`, `data/ipums/fetch_census_cps_org.py`) fetch data directly from official IPUMS and Census endpoints.

#### 3. Reassessing the Rising Immigrant Intermarriage Earnings Premium (1980–2024): A Replication Note and 14-Year Extension
*(Recommended reading order: 1st in family)*
- **Directory:** [`papers/paper2_intermarriage_comment/`](studies/economics/us-immigrant-integration/papers/paper2_intermarriage_comment/)
- **Manuscript:** [PDF](studies/economics/us-immigrant-integration/papers/paper2_intermarriage_comment/paper2_manuscript.pdf) · [LaTeX](studies/economics/us-immigrant-integration/papers/paper2_intermarriage_comment/paper2_manuscript.tex)
- **Summary:** Immigrants who arrive in the U.S. as children and later marry U.S.-born spouses earn higher hourly wages than comparable immigrants married to foreign-born spouses. Replicating Furtado and Song (2015) after removing U.S. citizens born abroad to American parents and extending the data 14 years through 2024, this study confirms that men's wage premium climbed from near zero in 1980 to a peak of about 12% in 2012–2015 before leveling off around 10% through 2024. Earlier work credited rising rewards for interpersonal and English communication skills, but comparing job tasks head-to-head shows that sorting into math- and analysis-heavy occupations explains far more of the trend than social or English tasks (45% vs. 14%).

#### 4. Generative AI Exposure and the Native–Immigrant Wage Gap: Evidence Across Exposure Measures in U.S. Wage Surveys (2015–2026)
*(Recommended reading order: 2nd in family)*
- **Directory:** [`papers/paper1_genai_immigrants/`](studies/economics/us-immigrant-integration/papers/paper1_genai_immigrants/)
- **Manuscript:** [PDF](studies/economics/us-immigrant-integration/papers/paper1_genai_immigrants/paper1_manuscript.pdf) · [LaTeX](studies/economics/us-immigrant-integration/papers/paper1_genai_immigrants/paper1_manuscript.tex)
- **Summary:** Controlled experiments and freelancing studies suggest tools like ChatGPT and Claude help non-native English speakers close writing gaps—but has that translated into higher relative pay across the U.S. workforce? Using 6.55 million worker records from the American Community Survey (2015–2024) and quarterly Current Population Survey earnings through late 2026, this study compares native-born and immigrant wages across jobs using both theoretical AI-exposure scores and observed workplace Claude usage. So far, it finds no detectable narrowing of the wage gap in writing-heavy or AI-exposed occupations, with sample sizes large enough to rule out even very small shifts.

#### 5. Ethnic Enclaves, Household Crowding, and Post-2020 Gateway Out-Migration: A Re-Examination (1980–2024)
*(Recommended reading order: 3rd in family)*
- **Directory:** [`papers/paper3_enclaves_wfh/`](studies/economics/us-immigrant-integration/papers/paper3_enclaves_wfh/)
- **Manuscript:** [PDF](studies/economics/us-immigrant-integration/papers/paper3_enclaves_wfh/paper3_manuscript.pdf) · [LaTeX](studies/economics/us-immigrant-integration/papers/paper3_enclaves_wfh/paper3_manuscript.tex)
- **Summary:** Why do U.S.-born descendants of immigrants keep living in ethnic enclaves even when immigration pushes up citywide housing costs? Using 17.7 million Census and ACS records (1980–2024) linked with FHFA and Zillow housing data to revisit Song and Xu (2020) and Song (2017), this study arrives at three findings. First, immigrant-dense neighborhoods show no clear rent or home-price discount inside metro areas; instead, households absorb high housing costs through denser living—more people per room and more three-generation homes—while earlier dynamic price regressions suffered from short-panel statistical bias. Second, the post-2020 remote-work boom did not spark a lasting exodus out of expensive gateway cities beyond a brief 2021 blip. Third, stricter causal checks confirm that U.S.-born descendants moving across states still reliably cluster near co-ethnic immigrant communities, while tests of local wages and resident retention prove inconclusive.
