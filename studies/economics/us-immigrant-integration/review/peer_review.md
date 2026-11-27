# Referee report: three papers extending Song (2017)

The three papers reviewed are under `papers/`:

- **P1:** GenAI and the native–immigrant wage gap
- **P2:** Intermarriage premium comment
- **P3:** Enclaves and WFH

Reviewed on 2026-11-24. I read every manuscript in full and read the code that produces each headline claim. I checked the headline numbers against `results/`. For P2 I also re-estimated the main specification and two placebo mediators from the saved parquet (script: `scratchpad/p2_check.py`).

**Overall:** The data are real and the numbers reproduce. I found no fabrication. The main problems are identification and inference, and in places the text says more than the code's own outputs support. None of the three is publishable in its current form. P2 is the closest, as a replication note.

## Cross-cutting

1. **The data are genuine.**
   - The CPS comes from Census public-use monthly files and is missing October 2025, which fits the 2025 shutdown.
   - The AEI files exist upstream at `Anthropic/EconomicIndex/labor_market_impacts/`.
   - IPUMS extracts are present.
   - My re-run of P2's upgraded specification reproduced the paper: 0.116 in 2012–15 and 0.092 in 2021–24.
2. **Forking paths.** The same pipeline produced opposite headlines across versions:
   - `REPORT.md` said "GenAI compresses the gap (+0.0123***)" and "WFH weakens enclave retention (−0.042pp**)".
   - The papers now report nulls.
   - P1 got its null by switching to a new exposure index after the old one gave a positive result.

   Each paper runs dozens to hundreds of specifications, with no pre-registration and no multiple-testing correction. Readers will discount marginal results heavily.
3. **AI authorship.** "Gemini 4 Neon" is listed as a co-author. Most publishers do not allow AI authors: Elsevier, Springer Nature, Wiley, T&F and the AEA journals do not, and ICMJE/COPE guidance agrees. It has to become a methods/acknowledgment disclosure, or desk rejection is likely.
4. **The prose oversells.** Examples:
   - "dead-zero", "precisely zero" (when the SE is 0.006)
   - "resolves the decades-old puzzle"
   - "causally accelerated"
   - "Killer Origin-Placebo" (in a code comment)

   Introductions run over 1,000 words and are dense with numbers. The writing is fluent but reads like advocacy rather than measured reporting.
5. **Silent fallbacks in the code.** Both P1's `fit_twoway_with_fallback` and P2's `run_feols` quietly drop two-way clustering to one-way (P2 can go all the way to heteroskedastic-robust) when a dimension has fewer than 15 clusters or the SEs come out NaN. The papers state two-way clustering throughout.
   - I checked P2's race subsamples: every one has at least 46 origin clusters and 406 metro clusters, so the 15-cluster trigger never applies there. I did not check whether the NaN trigger fired.
   - The fallback should still be logged, or removed. Separately, some Hispanic SEs (e.g., 0.0024) look implausibly small.
6. **Fairness to Song.** P3 calls Song (2017), an unpublished dissertation chapter, an "influential study". P1 says Song "documented" that computerization widened the gap. Yet the project's own audit (`REPORT.md`) found that result fragile: p = 0.06–0.09, and only with fixes. Represent the original accurately in both directions.

---

## Paper 1: "Writing Assist or Skill Equalizer?"

**Question:** Did generative AI narrow the native–immigrant wage gap? The design is an origin-placebo triple difference in the ACS (2015–24) and CPS ORG (2020Q1–2026Q3).

### Verified in code and results
- The headline CPS numbers match `aei_2026_cps_did_persistence.csv`: Eng 2025–26 = +0.0286 (p=.004) and DDD = −0.0245 (p=.040).
- **HonestDiD is hand-rolled** (`p1_acs_analysis.py` around L250). It is not the Rambachan–Roth package. It takes a plug-in worst-case bias, using the max pre-period step times the post-horizon, and applies a folded-normal critical value.
  - This is neither RR's ΔRM FLCI nor its conditional/hybrid intervals.
  - In the ACS, the uneven years (2015, 2018, 2019, 2021) are treated as equally spaced "steps".
  - Calling these "exact Rambachan–Roth" intervals is inaccurate.
- **Results the text omits or contradicts:**
  - In the CPS, *face-to-face social* exposure gives non-English-origin immigrants significant gains: +0.0145 (p=.002) in 2023–24 and +0.0115 (p=.03) in 2025–26, with DDD = +0.039 (p=.002). These estimates are in `tab2_cps_quarterly_did.tex`. The text never discusses them, and instead says social skills "continue to favor native English speakers", citing only the ACS.
  - The English-origin "windfall" is **negative** in 2023–24 (−0.0062) and turns positive only in 2025–26. That makes the 2023–24 DDD +0.0085, the opposite sign of the headline.
  - In the ACS, the English-origin gain loads on *coding* (+0.0094***), not writing. The mechanism story the paper tells cannot be checked against this.
- **ACS pre-trends.** The event study for observed exposure rises steadily before the "treatment": −0.010 (2015), −0.007, −0.003, −0.002, then 0 (2022), +0.002, −0.001. The pooled 2015–24 estimate of +0.0046** is pre-trend.

### Main concerns
1. **The exposure measure is post-treatment, single-provider and zero-inflated.**
   - The March 2026 Claude usage index is measured *after* treatment.
   - It reflects one provider's customer mix.
   - It is zero for 54% of occupations.

   Attenuation toward zero is the expected result of using a noisy, zero-inflated regressor, so a null from it is weak evidence. In the head-to-head regression the legacy index keeps +0.0074***. That contradicts the "phantom" interpretation: the legacy index predicts something the observed index misses.
2. **The placebo logic is inverted.** The English-origin group was supposed to show *no* effect. Finding a large effect means the placebo fails: UK, Canadian and Australian immigrants are concentrated in high-skill finance and tech. That is a violation of the comparison, not evidence against the writing hypothesis. With about 1,565 English-origin CPS workers in 2025–26, and the ACS DDD insignificant (p=.13), the abstract's "2.5 pp" claim is not robust.
3. **Power (CPS only).** "No equalization" needs a minimum detectable effect. A CPS SE of 0.006 per SD of exposure cannot rule out economically meaningful compression. The ACS SEs (0.0010–0.0017) are tight, so the ACS null is informative within its window.
4. **Robot channel.** "Echoes Song (2017)" overreaches. The robot coefficient is a 2015–24 trend with no break around 2022, and the "robot exposure" index is essentially a manual-occupation index.
5. **Minor:**
   - "81 consecutive months" is false because October 2025 is missing.
   - 2025Q4 has only two months.
   - The legacy estimate appears as 0.0049 and 0.0048; 0.0054 is the event-study average, a different estimand, and should be labelled as such.

### Novelty
Moderately novel. I found no US paper on GenAI and the native–immigrant wage gap specifically. Related work:
- Koo (2025, *Sci. Rep.*) is cross-lingual and experimental.
- IZA DP 18215 (2025) studies native/non-native convergence in academic writing.
- Aggregate GenAI wage nulls are well established: Humlum & Vestergaard (NBER w33777), the Finland register study (AEL 2025) and Korea (AEL 2026).

So the question is open, but a null is the expected prior, and the paper needs a credible design to add value.

**Verdict: reject in current form.** With a pre-period-measured exposure index, honest power and minimum-detectable-effect calculations, the actual RR package, full reporting of contrary estimates, and reframing as "no detectable equalization so far", it could be a major revision at a field journal (*Labour Economics*, *J. Population Economics*, *ILR Review*). It is not top-5 or top-field material.

---

## Paper 2: Reassessing Furtado & Song (2015)

**Question:** A replication, plus an extension to 2024, of the earnings premium for 1.5-generation immigrants married to natives.

### Verified
- **The replication is real.** My own re-estimate of the upgraded specification gives 2012–15 = 0.1163 and 2021–24 = 0.0916, matching the paper. The inverted-U holds.
- **Table 4 overlap: the finding is real, but its consequences are misstated.** White-identifying Hispanics are counted in both subsamples (46% of "White"), which is a fair point. But removing them from the White group cannot change the Black or Asian subsamples. In `forensic_table4_race.csv`, for 1980–2010:
  - NH Black 2008–10 increment = +0.081 (p=.13)
  - NH Asian increment = +0.070 (p=.15)

  Both are insignificant, exactly as Furtado & Song reported. The NH Asian *increment* stays insignificant through 2024 (+0.037, p=.45). The paper's "0.151*** peak" is a *level*, and the 1980 level was already 0.077. The abstract, Figure 1 caption and conclusion attribute the Black/Asian gains to the correction, and that is **false**: the correction contributes nothing to them. Section 3.3 does partly credit the 14 extra years of data. The Black result (+0.154***) appears only in 2021–24, with N ≈ 10k.
- **Table 3 "inversion".** The original *text* says "all but the college graduate category" where the table shows Some College is the insignificant one. That is a one-word typo; the tables are correct. Calling it a "forensic error in the original tables" overstates it. "College grads had the steepest surge" comes from the authors' new estimates, not from fixing the typo.
- **Sample definition.** `CITIZEN == 1` (born abroad to American parents, i.e., citizens at birth) is kept as "immigrant". That is 12.3% of men, and 81.6% of them have a native spouse. Dropping them barely moves the results (2012–15: 0.115, 2021–24: 0.094), so this flaw is real but immaterial. These men are also coded as "citizens" in the citizenship mediator.
- **The mediation placebo breaks the communication interpretation.** Script: `results/p2_review_check.py`. My base specification includes own-education controls, so its 2000→2021–24 rise is 0.022, not the paper's Gelbach base of 0.043. That makes the percentages below differ from the paper's 63.6%. I added each index × period separately and looked at how much of the rise remains:

  | Index added × period | Remaining rise |
  |---|---:|
  | Social + English (the paper's) | 0.0009 (~96% "explained") |
  | O\*NET math/analytical | 0.0042 (~81%) |
  | ALM abstract + RTI | 0.0034 (~85%) |

  The math/analytical index was built in `00_prepare_data.py` and never used.

  I also entered social, English and math **jointly**. The remaining rise is 0.0023, and the 2021–24 interactions are:

  | Index × 2021–24 | Coefficient | p |
  |---|---:|---:|
  | Social | **−0.056** | <.001 |
  | English | +0.076 | <.001 |
  | Math | +0.043 | <.001 |

  So the "social skills" channel actually turns *negative* once math is controlled for. Occupation is a post-treatment mediator, and any skill-intensity index absorbs the trend. The data cannot say "communication accounts for 63.6%"; the honest version is "occupational upgrading accounts for most of it".

### Other concerns
1. **The compression story is inconsistent.** It cites Autor–Dube–McGrew (bottom-of-distribution compression), yet the paper reports that *college* men drove the pullback.
2. **Possible geography break.** The 2012–15 peak coincides with the METAREA→MET2013 switch, which changes the geo×period fixed effects. This needs a consistent-geography robustness check.
3. **Few clusters.** With silent cluster fallback and around 20 Hispanic origin countries, the subgroup SEs are unreliable.
4. **Selection checks.** The IPW and Oster checks are well done and honestly caveated.

### Novelty
Low to moderate. It is a careful replication and extension; the new pieces are the post-2015 inverted-U, women, and the co-ethnic native-spouse split. It belongs in a replication or comment outlet:
- *J. Comments & Replications in Economics*
- a replication section such as *Labour Economics* or *Empirical Economics*
- a comment to *ANNALS*

**Verdict: major revision, as a replication note.**
- Drop the false Black/Asian "restoration" claim.
- Recast the Table 3 point as a text typo.
- Add placebo mediators and report "occupational sorting", not "communication".
- Fix the CITIZEN coding and the cluster fallback.
- Add a consistent-geography check.

It is the most solid of the three.

---

## Paper 3: "Enclaves Unbound?"

### Verified
- **Teleworkability is not pre-determined.** `03_module2...py` joins `tele_bin` on the respondent's *current* `OCCSOC`, which is observed after any move. The paper repeatedly calls it "pre-determined", "exogenous" and "causally" identified; none of those holds. It is ITT only if occupation does not respond to migration plans, and the paper itself says it might (§2.2).
- **Gateway-exit event study** (`module2_telework_event_study.csv`):

  | Year | Coefficient | p |
  |---|---:|---:|
  | 2015 | −0.11 pp | |
  | 2018 | **+0.25 pp** | |
  | 2021 | +0.53 pp | .001 |
  | 2022 | +0.11 pp | |
  | 2023 | +0.05 pp | |
  | 2024 | +0.23 pp | .14 |

  The 2018 pre-period coefficient is larger than the pooled post effect (+0.185 pp, p=.043). The whole result is one year, 2021. "Flat pre-trends" and "a second resurgence in 2024" are not supported.
- **The rent-burden "persistent climb"** (Panel D): the pooled DiD is significant (+0.00075, p=.017), but every event-study year is insignificant (p ≥ .17). That is too weak to call a "persistent climb".
- **"Enclave arbitrage" rests on very little.** `module3_enclave_arbitrage_regs.csv` has 42 regressions. The gateway-exit samples are clustered by origin metro, so there are **8 clusters**, and the p-values use t with 7 degrees of freedom:

  | Abstract claim | p |
  |---|---:|
  | +7.16% co-ethnic density | 0.079 |
  | −4.94 pp abandonment | 0.060 |

  Both are realized-WFH (endogenous) estimates. Section 5 and the limitations section do label them descriptive and note that the ITT estimates are null. The title, abstract and conclusion still lead with "enclave arbitrage", though. The arbitrage outcome itself, among high-enclave gateway exits, is **negative** (−0.0225). Gateway-exit Sunbelt probability is negative under both ITT and realized WFH: ITT −0.0097, WFH −0.041. Yet the abstract and conclusion say communities "shift toward affordable Sunbelt hubs". The 26–39% "rent savings" are descriptive means for any gateway leaver.
- **Nickell bias is asserted, not shown, in this paper.** Paper 3 never re-runs Song's lagged-price specification to show the bias. The earlier pass (`scripts/essay3_analysis.py`) re-estimated Song's Table 7, but that is not a Nickell demonstration either.
- **Arbitrary choices with no sensitivity analysis:** the top-8 gateway list, the 10% rent threshold and the 0.5 pp co-ethnic threshold.

### Other concerns
1. **Section 3 is static within-metro OLS with no instrument.**
   - PUMA-level share noise attenuates the rent "precise zero".
   - Controlling for rooms and bedrooms is a bad control when crowding is the proposed mechanism.
   - Crowding rising with immigrant share is partly compositional; the co-ethnic-native result is better, at t=2.2.
   - The claim to "resolve Saiz vs Song" ignores Saiz & Wachter (2011), which found *slower* neighborhood price growth in immigrant areas, and which the paper cites but does not engage.
2. **The JRS dynamic IV** uses waves 1980/90/00/10/19/24, so the lags are 10/10/10/9/5 years. It also mixes 5-year (Census) and 1-year (ACS) migration windows. The lag structure isn't comparable across waves, and the limitations section only partly concedes this.
3. **What holds up:** the AR/Sanderson–Windmeijer machinery is a genuine improvement, and the out-of-state-born split is a nice idea.

### Novelty
The remote-work × co-ethnic sorting question is relatively fresh; I found no close match. Immigration and crowding/doubling-up, and the donut effect, are established.

**Verdict: reject.** The causal language is unsupported by the treatment's timing, the headline mechanism fails its own ITT test and its arbitrage outcome, and inference rests on 8 clusters. A salvageable paper would need several things:
- Treat teleworkability measured before the move. Options: occupation one year ago is not in the ACS, so use the CPS/SIPP or the PSID, or industry-of-origin-metro exposure.
- Run wild-cluster bootstrap inference.
- Pre-specify the thresholds.
- Narrow the claim to a descriptive "who left gateways after 2020".

---

## Summary table

| | Science (design) | Validity (claims vs. evidence) | Writing / code | Novelty | Call |
|---|---|---|---|---|---|
| P1 | Weak: post-treatment, attenuated exposure; placebo misread | Several claims contradicted by its own CSVs | Fluent, oversold; hand-rolled "HonestDiD" mislabelled | Moderate (question open; nulls expected) | Reject; possible major revision at a field journal after reframing |
| P2 | Sound replication; mediation unidentified | One false headline claim (Black/Asian); Table 3 overstated | Reproducible; silent SE fallback; CITIZEN miscoding | Low–moderate | Major revision as replication note |
| P3 | Treatment not pre-determined; 8 clusters | Headline mechanism contradicted by own outputs | Same oversell; thresholds ad hoc | Moderate (WFH × enclaves) | Reject |

The code is real and useful, not misleading in the fabrication sense. But the manuscripts systematically pick favourable cells, and they describe properties of the methods ("pre-determined" teleworkability, "exact" Rambachan–Roth) that the code does not deliver.

## What I did not verify
- **Novelty searches.** I ran one search pass. Only P1's turned up close literature; the P2 and P3 searches returned adjacent work. Verify by hand before relying on the novelty calls.
- **Massenkoff & McCrory (2026).** I confirmed the dataset exists. I did not check whether their paper already estimates labor-market outcomes by observed exposure, which would bear on P1's novelty.
- **References.** Not systematically checked.
- **From memory, not checked here:**
  - which publishers ban AI authorship
  - the Saiz–Wachter (2011) neighborhood finding
  - the composition of the robot-exposure index
