# Peer review: Plateau: 3DEP LiDAR, prescribed fire, mesophication

Reviewed 2026-11-24. I read the manuscripts in full, then went into the code and outputs to check the headline claims. Each finding below is tagged:

- **[verified]**: I checked it in the repo's files or recomputed it from them.
- **[text]**: it rests on reading the manuscript alone.

None of the numbers I traced were fabricated. Every headline figure I followed (escarpment DiD, MODIS lapse, plateau BACI) matches the JSON/CSV outputs. A grep of the analysis scripts for synthetic, fake or fallback data found nothing that feeds results; I did not audit figure scripts for hard-coded values. The problems are in inference and framing, not in fabrication.

This file covers the Cumberland Plateau LiDAR & Fire study from the earth-science peer-review pass.

---

## 2. Plateau: 3DEP LiDAR, prescribed fire, mesophication

**Verdict: major revision. This is the strongest earth-science study.** The core design is good: repeat-LiDAR BACI, burn-unit clustering, a density-thinning check, and genuine unit tests on synthetic point clouds. The management prescriptions go beyond the evidence.

### Is it valid? (b)
1. **The +0.037 "oak canopy" coefficient can't be interpreted [verified formula].** In `lidar_ept.py:278`, `canopy_cover_10m = n_canopy / n_tot`, a share of all returns. Removing midstory returns raises it mechanically even if the canopy doesn't change, so a positive coefficient under fire is expected either way. The "intact canopy" claim may still be true (CHM p95 is unchanged), but this metric can't show it. Use first-return canopy cover, or a gap fraction at 10 m.
2. **Only the DiD is identified, not the control trend [text].** The 2012 QL3 and 2021 QL2 flights differ in sensor, scan angle and returns per pulse. Random thinning equalises density, not those. So "+0.113 densification in unburned stands" isn't identified; only the burned-minus-unburned contrast is. Daniel Boone (QL2 to QL2) is cleaner.
3. **The "5 m threshold" is a bin edge, and the pattern isn't monotone [text].** The four strata were chosen in advance, so 5 m was never estimated. In Table 3, one burn reduces 5–10 m density more (−0.023, p_SUID = 0.007) than 3+ burns do (−0.017, p_SUID = 0.16), which cuts against the escape story. The 3+-burn vs mech+fire difference is never tested directly.
4. **The mech+fire recommendation conflicts with the only before/after test of it [text].** The cross-sectional regression gives −0.067. But in the Daniel Boone BACI, mech+fire produced no reduction relative to controls (point estimate +0.015, n.s.). Mechanical thinning removing stems is also expected by design; that it beats fire alone on poles isn't a discovery.
5. **"Stopping fire causes rebound" isn't established [text].** p_SUID = 0.090, with n = 20 plots in few perimeters. Plots burned before 2017 started from a suppressed baseline, so faster regrowth is the expected recovery. To claim rebound past never-burned stands, you'd need to show 2023 *levels* exceed the controls'.
6. **Spring NDVI's advantage disappears within sites, and change was never tested [verified].** The pooled cross-sectional r of 0.41 (spring) vs 0.33 (winter) is mostly between-site. I recomputed from `plot_metrics.json`:
   - Within-site r: **spring 0.29 vs winter 0.28**.
   - Winter beats spring at Daniel Boone Main (0.52 vs 0.14) and Sewanee (0.55 vs 0.48).
   - `plot_metrics.json` stores spring NDVI for one period only (early/late splits exist only for winter), so ΔSpring NDVI against ΔLiDAR was never computed.

   To be fair, spring NDVI does respond to burn regime in the site-FE regression (Table 3), unlike winter. Yet Box 2 recommends spring NDVI for annual change tracking, which is the very standard on which winter NDVI was rejected. Also, `06_build_practitioner_explorer.py` labels this cross-sectional r as `BACI.spr_r`, which is misleading.
7. **The Landsat 5 "orbital and red-band drift" explanation is asserted without evidence.**

### Writing (c)
Clear and accessible, and the boxes are good. Tone down "proves", "pinpoints", "literally zero" and the prescriptive Box 2. Reframe Phase 1 as validating your own earlier method rather than overturning a field practice. That claim needs citations you don't have.

### Novelty
Moderate, and the strongest of the earth-science studies. Single-epoch LiDAR-versus-fire-frequency work exists (e.g., the 2024 ULS study in *Fire Ecology*), as do field burn trials (Arthur et al. 2015). A regional repeat-3DEP BACI on dated FACTS burns across three states is a genuinely useful contribution. Plausible venues after revision: *Fire Ecology* or *Forest Ecology and Management*.

---

## Summary

| Study | Good science? | Valid? | Writing/code | Novel enough? | Recommendation |
|---|---|---|---|---|---|
| Plateau (LiDAR/fire) | Yes, best design | Core BACI yes; canopy, spring-NDVI and prescription claims no | Clear; real unit tests | Yes, useful regional application | Major revision |

---

## Cross-cutting issues

1. **AI listed as an author.** "Gemini 4 Neon" is on the escarpment byline, and the papers say the work was "carried out by" agents. COPE, Nature, Science, Elsevier and Wiley do not allow AI authorship. Move this to a methods/acknowledgments disclosure, or desk rejection is likely.
2. **"Pre-specified" can't be verified, and was partly not honoured.** The gates were "locked" by agents, with no timestamped external registration. In `escarpment_study/profiling/` every script and output was written between 21:25 and 21:45 on Nov 23. Call these "internal decision criteria", not pre-registration. Where a gate failed, report it as failed (see escarpment 3 below).
3. **Each paper debunks its own earlier version as if it were the field's consensus.** Plateau: "researchers have widely monitored… winter NDVI" cites nobody; the foil is your own v1 (the README says so). Escarpment: "naive satellite regressions… exceed four days" is your own round-1 result (4.27 d/100 m in `log.md`). Reviewers will notice. Frame these as internal robustness checks, not overturned literature.
4. **Overprecision and pseudo-replication [text].** Values like p = 2.8×10⁻⁵⁸ and p = 2.2×10⁻⁴¹ (escarpment plateau-cap) come from pixels treated as independent. Report cluster/site-level inference, and cut most results to 2–3 significant figures.
5. **Leftover process artifacts.** Source comments like "STEP 4 OF STRICT WRITING ORDER", plus the `bestpaper-writing`/`check_draft.py` reports, should not ship with a submission.

---

*Limits of this review:* my literature search was targeted, not systematic. I did not re-run any Earth Engine extraction.
