# Three Empirical Studies on Immigrant Socio-Economic Integration in the United States

**Author / Director:** Daksh Tyagi  
**Provenance:** Directed by Daksh Tyagi; executed by a swarm of Gemini 4 Neon agents  
**Date:** November 2026  
**Discipline:** Labor & Urban Economics

This study family contains three open-science empirical papers re-examining and extending core questions in immigrant labor-market and residential integration in the United States, benchmarking against Furtado and Song (2015), Song (2017), Song and Xu (2020), and Song (2026).

All empirical scripts, aggregate regression outputs, LaTeX tables, figures, and compiled PDFs are published alongside an independent adversarial peer review ([`review/peer_review.md`](review/peer_review.md)) and a point-by-point response ([`review/response_to_referee.md`](review/response_to_referee.md)).

---

## Papers (Recommended Reading Order)

### 1. Reassessing the Rising Immigrant Intermarriage Earnings Premium (1980–2024): A Replication Note and 14-Year Extension
*Directory:* [`papers/paper2_intermarriage_comment/`](papers/paper2_intermarriage_comment/)

Using harmonized IPUMS U.S. Census and ACS microdata (1980–2024) linked to O\*NET and Autor–Dorn occupational task profiles, this replication note and 14-year extension reassesses Furtado and Song (2015, *The ANNALS of the American Academy of Political and Social Science*). Rebuilding the baseline on a clean foreign-born sample (`CITIZEN in {2, 3}`, excluding citizens born abroad to U.S. parents) with true full-metro working-age (18–64) enclave denominators reproduces the 1980–2010 rise (`0.095***` to `0.115***` in 2012–15) and shows a post-2015 plateau (`0.094***` in 2021–24; direct latest-minus-peak contrast `-0.021`, 95% CI `[-0.044, +0.003]`, `p = 0.084`). Competing occupational-task Gelbach decompositions show that while O\*NET Social + English tasks account for 57.7% of the 2000–2024 rise alone, O\*NET Math/Analytical tasks account for 54.4% alone and dominate in a joint horse race (`45.4%` vs. `14.2%`, with a negative direct `Social × 2021–24` return of `-0.058***`), pointing to broad occupational sorting rather than a uniquely communication-driven channel.

- **PDF:** [`papers/paper2_intermarriage_comment/paper2_manuscript.pdf`](papers/paper2_intermarriage_comment/paper2_manuscript.pdf) (17 pp.)
- **Source:** [`paper2_manuscript.tex`](papers/paper2_intermarriage_comment/paper2_manuscript.tex)
- **Tables & Scripts:** [`tables/`](papers/paper2_intermarriage_comment/tables/) · [`scripts/`](papers/paper2_intermarriage_comment/scripts/)

---

### 2. Generative AI Exposure and the Native–Immigrant Wage Gap: Evidence Across Exposure Measures in U.S. Wage Surveys (2015–2026)
*Directory:* [`papers/paper1_genai_immigrants/`](papers/paper1_genai_immigrants/)

This paper tests whether the late-2022 rollout of generative AI tools has yet produced detectable cross-occupation wage compression between non-English-origin immigrants and U.S.-born workers in national U.S. wage surveys. Combining individual-level IPUMS ACS microdata (2015–2024; `N = 6,546,097` workers, `3,840,043` in 2021–24) and top-code-harmonized Census Basic Monthly CPS ORG files (81 monthly files spanning Jan 2020–Oct 2026, excluding the Oct 2025 federal shutdown) across multiple occupational exposure measures—O\*NET 31.0 Written-English, Eloundou et al. (2024) OpenAI $\beta$, Felten et al. (2021/2023), and March 2026 Anthropic Economic Index observed usage—we find no consistent post-2022 equalization along the occupational-exposure wage gradient. Tight ACS nulls (`80% MDE = 0.0027–0.0047` log pts/SD) and noisier CPS estimates (`80% MDE = 0.0114–0.0171`) show that modest positive OpenAI $\beta$ estimates reflect pre-2022 upward drift and coding/math occupations rather than writing equalization.

- **PDF:** [`papers/paper1_genai_immigrants/paper1_manuscript.pdf`](papers/paper1_genai_immigrants/paper1_manuscript.pdf) (17 pp.)
- **Source:** [`paper1_manuscript.tex`](papers/paper1_genai_immigrants/paper1_manuscript.tex)
- **Tables & Scripts:** [`tables/`](papers/paper1_genai_immigrants/tables/) · [`scripts/`](papers/paper1_genai_immigrants/scripts/)

---

### 3. Ethnic Enclaves, Household Crowding, and Post-2020 Gateway Out-Migration: A Re-Examination (1980–2024)
*Directory:* [`papers/paper3_enclaves_wfh/`](papers/paper3_enclaves_wfh/)

This study re-examines the residential dynamics of U.S. ethnic enclaves across 1980–2024, benchmarking against Song and Xu (2020, *Southern Economic Journal*) and Song (2017, Ch. 3). Within-metro sub-metro panels of unique households (`PERNUM = 1`, `HHWT`-weighted) show flat rent and home-value gradients alongside clear absorption through household crowding (persons per room) and three-or-more-generation living (`MULTGEN = 3`), and a same-sample dynamic-panel check shows how lagged-level FE specifications induce severe Nickell bias ($\hat\rho_{\text{FE}} = -0.31$ vs. Anderson–Hsiao $\hat\rho_{\text{AH}} = -0.00$ for rent). Descriptive post-2020 inter-MSA mover regressions audited with Webb (2023) 6-point wild cluster bootstrap ($G = 8$ gateway clusters) find no consistent positive differential in Sunbelt relocation or high-enclave arbitrage among observed gateway leavers. Separating a uniform 10-year decadal shift-share panel (1980–2010) from a modern ACS extension (2010–2024) with closed-form analytical Anderson–Rubin 95% confidence sets confirms bounded positive out-of-state-born co-ethnic adult clustering (`[+0.013, +0.051]` decadal; `[+0.011, +0.041]` ACS).

- **PDF:** [`papers/paper3_enclaves_wfh/paper3_manuscript.pdf`](papers/paper3_enclaves_wfh/paper3_manuscript.pdf) (15 pp.)
- **Source:** [`paper3_manuscript.tex`](papers/paper3_enclaves_wfh/paper3_manuscript.tex)
- **Tables & Scripts:** [`tables/`](papers/paper3_enclaves_wfh/tables/) · [`scripts/`](papers/paper3_enclaves_wfh/scripts/)

---

## Transparent Peer-Review Audit

All three drafts underwent an independent adversarial, code-level peer review before release, followed by a point-by-point revision response and follow-up verification audit:

- **Adversarial Peer Review:** Full code- and manuscript-level critique in [`review/peer_review.md`](review/peer_review.md), leading to re-estimation of placebo mediators, Webb wild cluster bootstraps, consistent-geography checks, uniform-lag JRS panels, and explicit retractions of unsupported preliminary claims.
- **Response & Follow-Up Verification:** Point-by-point response by Gemini 4 Neon in [`review/response_to_referee.md`](review/response_to_referee.md), including follow-up code and equation checks (individual-vs.-cell ACS aggregation, full-metro enclave denominators, `PERNUM == 1`/`HHWT` vs. `PERWT` separation, `MULTGEN == 3`, same-sample Nickell checks, analytical 1D Anderson–Rubin sets, direct 2021–24 vs. 2012–15 contrasts, and portable paths).

---

## End-to-End Reproduction Instructions

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Acquire Public Microdata & Bulk Auxiliary Datasets

Per IPUMS Terms of Use, **raw and derived person-level IPUMS USA microdata are excluded from Git** (along with API keys and bulk raw O\*NET / housing dumps). Processed lightweight crosswalks and aggregate regression outputs are included.

1. **IPUMS USA Microdata (1980–2024):** Register for a free IPUMS USA account and API key at [usa.ipums.org](https://usa.ipums.org/), place the key in `data/ipums/.api_key` (gitignored) or export `IPUMS_API_KEY`, and run:
   ```bash
   python data/ipums/submit_extracts.py --submit --wait-and-download
   ```
2. **Census Basic Monthly CPS ORG (Jan 2020–Oct 2026):**
   ```bash
   python data/ipums/fetch_census_cps_org.py
   ```
3. **Auxiliary Occupational & Housing Sources:** See [`data/aux/README.md`](data/aux/README.md) for upstream O\*NET 31.0, FHFA HPI, Zillow ZHVI/ZORI, and AI exposure file layouts, then rebuild crosswalks if needed:
   ```bash
   python scripts/build_occ_tasks.py
   python scripts/build_housing_panel.py
   ```

### 3. Run Paper Pipelines

**Paper 2 (Intermarriage Premium):**
```bash
python papers/paper2_intermarriage_comment/scripts/00_prepare_data.py
python papers/paper2_intermarriage_comment/scripts/01_forensic_and_gelbach.py
python papers/paper2_intermarriage_comment/scripts/02_arrival_gender_ipw.py
python papers/paper2_intermarriage_comment/scripts/03_make_figures_and_tables.py
```

**Paper 1 (Generative AI & Immigrant Wages):**
```bash
python papers/paper1_genai_immigrants/scripts/p1_build_tasks.py
python papers/paper1_genai_immigrants/scripts/p1_acs_analysis.py
python papers/paper1_genai_immigrants/scripts/p1_acs_individual_check.py
python papers/paper1_genai_immigrants/scripts/p1_cps_analysis.py
python papers/paper1_genai_immigrants/scripts/p1_aei_2026_upgrade.py
python papers/paper1_genai_immigrants/scripts/p1_make_acs_figures_tables.py
python papers/paper1_genai_immigrants/scripts/p1_make_cps_figures_tables.py
python papers/paper1_genai_immigrants/scripts/p1_make_aei_2026_figures_tables.py
```

**Paper 3 (Ethnic Enclaves, Crowding & WFH):**
```bash
python papers/paper3_enclaves_wfh/scripts/01_build_onet_teleworkability.py
python papers/paper3_enclaves_wfh/scripts/02_module1_cpuma_neighborhoods.py
python papers/paper3_enclaves_wfh/scripts/03_module2_telework_did_event_study.py
python papers/paper3_enclaves_wfh/scripts/04_module3_enclave_arbitrage_matrix.py
python papers/paper3_enclaves_wfh/scripts/05_module4_jrs_weak_iv_robust.py
python papers/paper3_enclaves_wfh/scripts/06_make_figures_and_tables.py
```

### 4. Compile All Three PDFs Locally

Requires `tectonic` (or `pdflatex` + `bibtex`):

```bash
bash scripts/compile_papers.sh
```

---

## Citation

See [`CITATION.bib`](CITATION.bib) for individual BibTeX entries for all three papers.
