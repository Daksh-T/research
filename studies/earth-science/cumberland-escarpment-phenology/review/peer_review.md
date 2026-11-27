# Peer review: Escarpment: "Trapped Cold Air and Sandstone Oaks"

Reviewed 2026-11-24. I read the manuscripts in full, then went into the code and outputs to check the headline claims. Each finding below is tagged:

- **[verified]**: I checked it in the repo's files or recomputed it from them.
- **[text]**: it rests on reading the manuscript alone.

None of the numbers I traced were fabricated. Every headline figure I followed (escarpment DiD, MODIS lapse) matches the JSON/CSV outputs. A grep of the analysis scripts for synthetic, fake or fallback data found nothing that feeds results; I did not audit figure scripts for hard-coded values. The problems are in inference and framing, not in fabrication.

This file covers the Cumberland Escarpment study from the earth-science peer-review pass.

---

## 1. Escarpment: "Trapped Cold Air and Sandstone Oaks"

**Verdict: major revision.** There is a real, replicated cold-pool signal here, but it is smaller than reported, and several "verified" claims are weaker than stated.

### Is it good science? (a)
Mostly. Testing cold-air pooling with replicated closed-vs-open basin pairs, plus ECOSTRESS thermal data and a wind-collapse test, is a strong design. Honest null results (H6, H7, the Manski bounds) count in its favour.

### Is it valid? (b)
1. **Relief confound inflates the headline [verified].** Open-gorge controls have more floor-to-rim relief than the closed basins: 178 m vs 127 m mean (`profiling/p1.json`, `site_meta_30m`). Elevation alone pushes floor–rim differences in the hypothesis's favour. Adding relief as a covariate across the 22 sites (NDRE) gives:
   - ΔSOS: +7.1 → **+4.3 d** (SE 1.1)
   - ΔEOS: −4.1 → −4.8 d
   - ΔGSL: −11.1 → **−9.1 d** (SE 1.9)

   NDVI behaves the same way: GSL −10.7 → −8.0. The effect survives, but the abstract's "11.1 days" should be relief-adjusted, or the pairs re-matched on relief.
2. **"FillDepth explains 66%, TPI explains 0%" restates the group labels [verified].** FillDepth is 0 for every open site and ≥22 m for every closed one. The 20 m step model's dGSL coefficient is −11.14, identical to the paired DiD. The saturating hinge (β = −0.45 d/m, about −11.2 d at the 25 m cap) ties it (ΔAICc = 0.03). The paper does concede that tie and the unobserved 0–22 m range, but it still headlines the R² as evidence that closure geometry wins. It is the closed-vs-open contrast again, not a dose-response. Within closed basins, deeper sills do not senesce earlier: r(dEOS, FillDepth) = +0.24, p = 0.47. TPI explaining zero is a fair point; "closure geometry governs" via R² = 0.66 is not.
3. **The P1 gate failed its own criteria but is reported as "Verified" [verified].** The coded P1 rule required open gorges to show floor-minus-rim ΔEOS ≥ 0 and |r(σSOS, FillDepth)| < 0.15. Open gorges came in at −3.03 d, and r came in at 0.22 (NDVI). `scorecard.json` says `PARTIAL_PASS`, yet Table 4 lists H3 as "Verified", and §4.1 adds a post-hoc explanation for the failed criterion.
4. **The MODIS lapse holds up, but its ± is ambiguous [verified].** The ±0.82 is the interannual SD over 23 years (`v2_hopkins_climate_lal.json`), so SE ≈ 0.17. 3.19 vs 4.09 is therefore distinguishable, and the Hopkins match stands. But the paper uses ± for SD in some places (−9.28 ± 2.57, n = 11) and for SE in others (−1.55 ± 0.27, n = 36), without saying which. State it every time. Separately, MidGreenup (50%) and Greenup (15%) measure different things, so calling the 15% lapse an "artifact" is too strong. Onset and mid-green-up can legitimately lapse differently.
5. **The plateau-cap "guild turnover" is inferred, not confirmed [text].** There is no species data. mARI is an indirect proxy, and the fragipan null rests on smoothed modelled soils (POLARIS), so absence of evidence is weak. Say "consistent with".
6. **ECOSTRESS [text].** The thermal work and wind-collapse test are the best evidence in the paper. One caveat: the 7/7 cove result covers only one MGRS tile, and two coves have no p-value reported.

### Writing and code (c)
- The title says the law "Breaks", but the abstract and conclusion say it holds along open slopes. Pick one.
- The prose is very number-dense, and words like "decisively", "proving" and "crucially" appear throughout.
- `paper/writing_report.md` says limitations were "deliberately kept out of" the abstract and introduction. Reviewers expect the main caveats there, especially that the swath-level Manski bounds span zero.
- The AlphaEarth sidebar (§7, Table 6) belongs in a separate paper. Its LOCO comparison rests on 4 folds.
- `tests/test_v2_pipeline.py` mostly pins conclusions, e.g. `assertLess(dGSL, -8.0)` and `delta_aicc < -15`. These are not logic tests: a bug fix that shrank the effect would "fail" the suite. Replace them with unit tests on the fitting and pairing code.

### Novelty
Moderate to low. Mizuno & Nasahara (2026, *Theor. Appl. Climatol.*; I confirmed it exists) already show cold-air pools reversing elevational phenology, using HLS plus nighttime LST. The real addition here is replication across 11 matched karst basins, plus the closed-vs-open contrast. Your own `log.md` notes a prior satellite test of Hopkins' law ("Jiang 2022"). I haven't checked that reference, but it isn't cited. Find it and position against it.

---

## Summary

| Study | Good science? | Valid? | Writing/code | Novel enough? | Recommendation |
|---|---|---|---|---|---|
| Escarpment (karst/Hopkins) | Mostly | Signal real but smaller after relief (GSL −11.1 → −9.1 d; SOS +7.1 → +4.3 d); FillDepth/TPI and "verified" P1 overstated | Dense, overclaiming; tests pin results | Incremental over Mizuno & Nasahara 2026 | Major revision |

---

## Cross-cutting issues

1. **AI listed as an author.** "Gemini 4 Neon" is on the escarpment byline, and the papers say the work was "carried out by" agents. COPE, Nature, Science, Elsevier and Wiley do not allow AI authorship. Move this to a methods/acknowledgments disclosure, or desk rejection is likely.
2. **"Pre-specified" can't be verified, and was partly not honoured.** The gates were "locked" by agents, with no timestamped external registration. In `escarpment_study/profiling/` every script and output was written between 21:25 and 21:45 on Nov 23. Call these "internal decision criteria", not pre-registration. Where a gate failed, report it as failed (see escarpment 3 below).
3. **Each paper debunks its own earlier version as if it were the field's consensus.** Plateau: "researchers have widely monitored… winter NDVI" cites nobody; the foil is your own v1 (the README says so). Escarpment: "naive satellite regressions… exceed four days" is your own round-1 result (4.27 d/100 m in `log.md`). Reviewers will notice. Frame these as internal robustness checks, not overturned literature.
4. **Overprecision and pseudo-replication [text].** Values like p = 2.8×10⁻⁵⁸ and p = 2.2×10⁻⁴¹ (escarpment plateau-cap) come from pixels treated as independent. Report cluster/site-level inference, and cut most results to 2–3 significant figures.
5. **Leftover process artifacts.** Source comments like "STEP 4 OF STRICT WRITING ORDER", plus the `bestpaper-writing`/`check_draft.py` reports, should not ship with a submission.

---

*Limits of this review:* my literature search was targeted, not systematic. I did not re-run any Earth Engine extraction.
