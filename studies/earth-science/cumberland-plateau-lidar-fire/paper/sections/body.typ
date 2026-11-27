#let callout_box(title, body) = block(
  width: 100%,
  fill: rgb("#f3f6f4"),
  stroke: (top: 1.2pt + rgb("#1d4d3e"), bottom: 0.7pt + rgb("#c6d1cb"), left: 0.5pt + rgb("#d7dfda"), right: 0.5pt + rgb("#d7dfda")),
  inset: (x: 11pt, y: 9pt),
  radius: 2pt,
  breakable: false,
)[
  #text(weight: "bold", size: 9.5pt, fill: rgb("#163b30"))[#title]
  #v(4pt)
  #set text(size: 9.1pt)
  #set par(leading: 0.58em, justify: true)
  #body
]

= Background and Previous Work

Three lines of research converge on eastern upland oak--hickory (_Quercus_--_Carya_) forests of the Cumberland Plateau: field fire ecology, passive satellite monitoring, and active three-dimensional laser scanning.

Field experiments provide the ecological baseline. For centuries, periodic low-intensity surface fires kept upland oak forests relatively open underneath, allowing sun-loving oak seedlings to recruit while suppressing thin-barked competitors. Following twentieth-century fire exclusion, shade-tolerant hardwoods (notably red maple, _Acer rubrum_; American beech, _Fagus grandifolia_; and blackgum, _Nyssa sylvatica_) expanded into the midstory beneath mature oaks @Nowacki2008 @Hart2012 @Knott2019 @Alexander2021. Because these species cast deeper shade and produce moister, faster-decomposing leaf litter, they cool and dampen the forest floor in a self-reinforcing process termed _mesophication_. Multi-year prescribed-burn trials in Kentucky's Daniel Boone National Forest showed that dormant-season surface fires top-kill small seedlings and saplings, whereas larger, thicker-barked midstory stems survive mild surface flames unless fires are repeated or paired with mechanical thinning @Arthur2015. Field trials yield species-level detail but cover small experimental footprints; scaling structural evaluation across hundreds of kilometers requires remote sensing.

Passive satellite imagery offers wall-to-wall coverage over four decades, making it tempting to look for an optical shortcut. #cite(<Chastain2007>, form: "prose") demonstrated that leaf-off winter Landsat greenness (the Normalized Difference Vegetation Index, or NDVI) maps _evergreen_ subcanopy shrubs such as mountain laurel (_Kalmia latifolia_) and great rhododendron (_Rhododendron maximum_) beneath leafless Appalachian hardwoods. Extending that leaf-off NDVI approach to _deciduous_ mesophication (as we initially attempted in an exploratory unharmonized workflow on Tennessee's Cumberland Plateau) appears at first glance to show a steep four-decade winter greening trend. Long Landsat composites also frequently rely on cross-sensor adjustment equations from #cite(<Roy2016>, form: "prose") to link older and newer sensors. Here we test directly whether leaf-off or early-spring Landsat indices can actually track deciduous subcanopy structure and prescribed-fire change in Collection 2 archives.

#v(4pt)
#callout_box("Box 1. A Plain-Language Guide to Key Terms in This Study")[
  - *Mesophication:* The gradual filling-in of an oak forest after fire is excluded. Shade-tolerant trees like red maple and beech crowd beneath the oaks, making the forest floor shadier, damper, and harder for oak seedlings or surface fires to persist.
  - *NDVI (Satellite Greenness):* An index from $-1$ to $+1$ comparing near-infrared and red sunlight reflected by land. Green foliage produces high NDVI; leafless winter tree trunks and branches do not.
  - *Airborne LiDAR, Midstory NRD, and First-Return Canopy:* Aircraft fire rapid laser pulses down through bare winter treetops. *Normalized Relative Point Density (NRD)* divides hits inside a height band by all hits reaching that band's ceiling, adjusting for upper-canopy shading. To measure the $>10$ m overstory without mechanical distortion from midstory removal, we use *first-return canopy cover*: the fraction of initial laser strikes (`return_number = 1`) intercepted above 10 m.
  - *Repeat-Scan Comparison (BACI / DiD):* Short for Before--After--Control--Impact. Scanning the same burned and unburned forest patches years apart strips away fixed terrain differences. Because sensor hardware and multi-return sensitivity differ across flights, only the *difference-in-differences (DiD)*, meaning burned change minus unburned change, is causally identified.
  - *Burn-Unit Clustering ($p_"SUID"$):* Multiple plots inside one prescribed burn share ignition and weather conditions. Clustering standard errors by official burn-unit ID (`suid`) prevents spatial pseudo-replication from overstating confidence.
]
#v(4pt)

Active laser scanning (LiDAR) measures woody forest geometry directly rather than relying on reflected leaf greenness. NASA's spaceborne GEDI mission samples vertical plant area profiles along orbital tracks @Dubayah2020, while single-epoch airborne and unoccupied laser scanning studies have begun linking fire frequency to understory structure. The U.S. Geological Survey's 3D Elevation Program (3DEP) now provides public, wall-to-wall winter airborne LiDAR across much of the eastern United States, including repeat acquisitions six to nine years apart. Here we combine a radiometric audit of 40-year Landsat Collection 2 archives with 689 pure-deciduous 3DEP plots across three states, two repeat-LiDAR BACI experiments paired with dated official burn records, unbiased first-return canopy metrics, continuous 1-meter vertical height profiles, and head-to-head temporal and cross-sectional validation of Landsat and GEDI against aircraft LiDAR.

= Study Region and Methodology

== Study Transect and Authoritative Fire Records

We evaluate optical proxies and three-dimensional forest structure across a 480 km transect of the Cumberland Plateau spanning Alabama, Tennessee, and Kentucky (@fig-map). First, for the 40-year Landsat radiometric audit, we analyze the southern Tennessee plateau upland ($>= 430$ m elevation; 5,617 km#super[2] forested in 1985) and eight protected areas: Sewanee Domain (5,190 ha), Franklin State Forest (3,564 ha), Savage Gulf State Natural Area (6,896 ha), Prentice Cooper State Forest (10,009 ha), Catoosa Wildlife Management Area (32,048 ha), Bridgestone/Firestone Centennial Wilderness (4,915 ha), North Chickamauga Creek Gorge (2,885 ha), and Fall Creek Falls State Park. Because standard WDPA exports represent Fall Creek Falls only by its 2,946 ha WMA sub-unit, we use the complete 10,282 ha OpenStreetMap state park boundary.

#figure(
  image("../figures/fig0_study_area_map.png", width: 82%),
  caption: [*The Cumberland Plateau study transect across Kentucky, Tennessee, and Alabama.* Elevation and shaded relief derive from Google Earth Engine SRTM terrain data (150--750 m a.s.l.). Colored points locate all 689 pure-deciduous 3DEP airborne LiDAR plots across Bankhead National Forest (AL, $n = 275$), Sewanee Domain and Franklin State Forest (TN, $n = 79$), and Daniel Boone National Forest (KY, $n = 335$), alongside the eight Tennessee protected areas and regional optical audit window (WGS 84 / EPSG:4326).],
) <fig-map>

Second, to test structural responses to prescribed fire, we sample three plateau landscapes that combine mapped, dated prescribed-burn records with winter leaf-off 3DEP airborne LiDAR:

1. *William B. Bankhead National Forest, Alabama (southern plateau):* 565 U.S. Forest Service (USFS FACTS) completed broadcast-burn perimeters (2007--2026) and 226 mechanical fuel-treatment polygons. In northern Bankhead, March 2012 (`AL_LawrenceCo_2012`, QL3) and February 2021 (`AL_17Co_1_2020`, QL2) flights create an 8.9-year repeat-scan BACI dataset ($n = 139$). Southern Bankhead adds 136 cross-sectional plots scanned in January 2017 (`AL_25Co_B1_2017`, QL2).
2. *Daniel Boone National Forest, Kentucky (northern plateau):* 461 completed USFS broadcast-burn polygons (2006--2026) and 1,602 mechanical-treatment polygons. A statewide winter flight (`KY_FullState`, primarily February--March 2017, QL2) provides 189 main-forest plots and pairs with a February 2023 flight (`KY_CentralEast_3_A23`, QL2) for a 6.0-year QL2-to-QL2 BACI dataset in eastern Daniel Boone ($n = 146$ sampled; $n = 144$ verified on a uniform 2017 baseline after removing two controls whose baseline tile dated to December 2012).
3. *Sewanee Domain and Franklin State Forest, Tennessee (central plateau):* 84 mapped prescribed-burn polygons from Sewanee's Landscape Analysis Lab (2001--2026) paired with unburned adjacent Franklin State Forest under a January 2016 flight (`TN_27County_blk4_2015`, QL2; $n = 79$).

== Phase 1: Landsat Collection 2 Radiometric Audit

To test whether the apparent 40-year winter greening trend reflects ecological change or sensor discontinuity, we processed Landsat Collection 2 Tier 1 Level-2 scenes (Landsat 5 TM, 7 ETM+, 8 OLI, and 9 OLI-2) with strict `QA_PIXEL` masking of cloud, cloud shadow, cirrus, and snow (bits 1--5) and reflectance bounds $(0, 1)$. Annual median composites were constructed for winter (December--March and core January--February), spring green-up (March--April), peak summer (June--August), and autumn (October--November), and long-term pixel slopes were estimated via Theil--Sen regression (`sensSlope`, reported per decade).

Because consecutive Landsat sensors overlapped in orbit, we measured pixel-matched inter-sensor offsets directly on intact hardwood forest during overlapping calendar years (Landsat 7 minus 5 in 2000--2011; Landsat 8 minus 7 in 2013--2021). We compared four trend specifications: (1) uncorrected Collection 2, (2) the Collection 1 reduced major axis adjustment of #cite(<Roy2016>, form: "prose") Table 3, (3) empirical overlap harmonization anchored to Landsat 7, and (4) single-sensor eras (Landsat 5 in 1985--2011, Landsat 7 in 2000--2021, and Landsat 8+9 in 2014--2025). To evaluate the residual single-sensor Landsat 5 trend (1985--2011), we stratified plateau forest by Annual NLCD (1985, 2001, 2021) into old deciduous, regrown deciduous, old evergreen, and regrown evergreen cohorts, and cross-checked winter NDVI against shortwave-infrared indices (NBR, NDMI) and Daymet V4 winter temperature and snow records.

== Phase 2: Pure-Deciduous Sampling and 3DEP LiDAR Metrics

To avoid confusing deciduous mesophication with pine plantations or timber harvests, every 30 m radius circular plot (0.28 ha; spaced $>= 220$ m apart in national forests and $>= 180$ m on Sewanee) was screened in Google Earth Engine before LiDAR extraction. Eligible plots required $>= 80\%$ neighborhood purity of continuous Annual NLCD Class 41 deciduous forest in 2001, 2010, and 2021, Hansen `treecover2000 >= 60%`, and zero stand-replacing loss through 2025. Stands overlapping USFS FACTS mechanical-treatment polygons were isolated into a separate `mech_plus_fire` category. Topographic covariates (elevation, slope, Heat Load Index or HLI, and 450 m Topographic Position Index or TPI) were extracted on the native 10 m USGS elevation grid (`USGS/3DEP/10m_collection`), yielding 689 plots.

#figure(
  image("../figures/fig_mechanism_schematic.png", width: 98%),
  caption: [*Sensing physics and vertical response profile across height strata.* (a) Passive Landsat optical indices miss temporal structural change beneath leafless oak crowns (BACI winter $r = -0.00$, spring $r = -0.05$), whereas leaf-off 3DEP airborne LiDAR resolves woody stems via penetration-adjusted NRD and unbiased first-return upper canopy cover ($>10$ m). (b) Across 1-meter vertical bins, repeated surface fire reduces stem density most strongly from 1 to 4 m, attenuates across a 4--7 m crossover zone, and leaves first-return $>10$ m canopy statistically unchanged, whereas cross-sectional mechanical thinning + fire is stronger at 8--10 m and opens the $>10$ m canopy.],
) <fig-schematic>

Raw 3DEP Entwine Point Tile (`.laz`) clouds were streamed directly from AWS S3 (`usgs-lidar-public`). After discarding building, water, and noise returns (LAS Classes 6, 7, 9, 18) and decoding GPS timestamps to verify leaf-off acquisition dates, ground elevation was interpolated from LAS Class 2 returns using 6-nearest-neighbor inverse-distance weighting across a 6 m buffer, and every return inside the 30 m true-ground radius (correcting Web Mercator latitude distortion) was converted to height above ground (HAG).

To control for upper-canopy occlusion, we compute Beer--Lambert penetration-adjusted Normalized Relative Point Density (NRD) for any vertical band $[z_1, z_2]$:

$ "NRD"(z_1, z_2) = (N(z_1 <= "HAG" < z_2)) / (N("HAG" < z_2)) $ <eq-nrd>

We evaluate four ecological bands: understory sprouts and shrubs (0.5--2.0 m), low midstory saplings (2.0--5.0 m), tall midstory poles (5.0--10.0 m), and combined midstory (2.0--10.0 m). We also compute a continuous *1-meter vertical NRD profile* (`0.5--1 m`, `1--2 m`, ..., `9--10 m`) so we can inspect how treatment differences vary smoothly with height rather than relying solely on pre-selected bin edges.

Measuring the mature $>10$ m overstory requires care. Computing canopy cover as a share of _all_ returns ($N("HAG" >= 10) / N_"total"$) is mechanically biased under prescribed fire: removing subcanopy returns shrinks $N_"total"$ and inflates the $>10$ m share even when the upper canopy does not change. We therefore measure overstory cover using *first-return canopy cover* (`first_return_canopy_10m` $= N("return" = 1, "HAG" >= 10) / N("return" = 1)$) and its complement, 10 m canopy gap fraction, alongside 95th-percentile vegetation height (`CHM p95`) and Foliage Height Diversity (FHD across 2 m bins). Removing a stem below 10 m cannot create a first return above 10 m, eliminating mechanical coupling; we verify this property explicitly on synthetic point clouds in our automated test suite.

== Statistical Identification and Sensor Validation

Because prescribed burns are often sited on drier, gentler terrain than unburned coves, we combine three complementary analyses:

1. *Topographic Matching:* Within each site, burned plots are matched 1:1 to unburned controls via regularized covariance-weighted Mahalanobis distance across elevation, slope, HLI, and 450 m TPI, evaluated with paired $t$-tests, Wilcoxon tests, Cohen's $d$, and 4,000-iteration bootstrap confidence intervals (flagging when limited controls require replacement).
2. *Pooled Fixed-Effects OLS with Burn-Unit Clustering and Direct Contrasts ($n = 689$):* We regress each vertical metric (including all ten 1 m height bins and first-return canopy cover) and optical index on treatment indicators (`1`, `2`, `3+` burns, and `mech_plus_fire`), standardized terrain covariates, point density, and site fixed effects. Alongside White HC3 standard errors, we compute Liang--Zeger CR1 cluster-robust standard errors grouped by official burn-unit ID (`suid`, $G = 359$) and 2 km spatial blocks ($G = 377$), test the direct linear contrast `mech_plus_fire` minus `burn_3plus` using the full clustered covariance matrix, and check robustness after dropping 9 cross-site plots situated within 220 m of a neighboring site ($n = 680$).
3. *Epoch-Verified Multi-Epoch BACI Differencing ($n = 283$):* On Bankhead North (2012 to 2021, $n = 139$) and Daniel Boone East (2017 to 2023, $n = 144$), we compute plot-level change ($Delta "NRD" = "NRD"_2 - "NRD"_1$) and verify exact USFS FACTS `date_completed` timestamps for both fire and mechanical polygons strictly between decoded LiDAR flight dates (`acq_baseline < date_completed <= acq_primary`). Crucially, we track return-density and first-return-fraction shifts across flights, randomly thin the denser cloud on every plot to match the sparser flight's density, and treat *only the burned-minus-unburned DiD* (not the raw control trend) as causally identified. Alongside CR1 burn-unit clustering and 2 km spatial blocks, we compute restricted Rademacher wild cluster bootstrap $p$-values (`WCR-t`, 9,999 draws, $p_"WCB"$) to guard against small-cluster bias when treated arms contain only 3--4 burn units. We also fit baseline-adjusted ANCOVA models and compare baseline and endpoint levels against controls (for both 2--10 m midstory and first-return $>10$ m canopy) to separate genuine treatment effects from baseline recovery.

On the same 689 plots, we extract Landsat 8/9 seasonal indices (winter Jan--Feb NDVI/NBR/NDMI; spring Mar--Apr, summer Jun--Aug, and autumn Oct--Nov NDVI), early (2014--2016) to late (2020--2023) Landsat OLI deltas for *both* winter and spring (plus 2016--2018 to 2022--2024 spring deltas aligned to Daniel Boone's 2017--2023 interval), and quality-filtered NASA GEDI L2B vertical PAI (2019--2023, 75 m radius, $n = 359$). To distinguish between-site gradients from true stand-level skill, we compute both *pooled* and *within-site (site-demeaned)* correlations alongside temporal BACI delta correlations.

= Results

== Phase 1: Inter-Sensor Jumps Dominate 40-Year Winter Landsat Trends

Our radiometric audit shows that most of the apparent 40-year winter NDVI increase in an unharmonized Landsat Collection 2 workflow is an inter-sensor calibration step (@fig-optical a and @tab-optical). During overlapping calendar years on identical pure-deciduous plateau pixels, Landsat 8 OLI reads $+0.093$ NDVI higher in January--February than Landsat 5 TM ($+0.024$ from L5 to L7, plus $+0.069$ from L7 to L8). In summer, when full leafy canopies saturate near 0.86--0.90 NDVI, the L5-to-L8 step is less than half as large ($+0.040$), creating the false appearance that winter greening outpaced summer greening.

#figure(
  image("../figures/fig1_optical_illusion_vs_lidar.png", width: 99%),
  caption: [*Landsat optical trends and BACI delta performance relative to 3DEP LiDAR.* (a) Removing empirical Collection 2 inter-sensor offsets reduces the 40-year plateau winter NDVI slope by 79%, and modern Landsat 8+9 (2014--2025) slopes downward. (b) Across repeat-scanned BACI plots, neither winter ($r = -0.00, p = 0.98, n = 272$) nor spring ($r = -0.05, p = 0.37, n = 283$) Landsat NDVI change tracks measured 3D midstory change. (c) Cross-sectional pooled FE-OLS estimates across 689 plots (95% burn-unit clustered CIs) show burn-regime responses in 3DEP LiDAR and spring NDVI while winter NDVI remains flat.],
) <fig-optical>

#figure(
  kind: table,
  caption: [*Regional Cumberland Plateau ($>= 430$ m elevation) Landsat Collection 2 winter NDVI trends across calibration methods and forest-age cohorts.*],
  table(
    columns: (2.05fr, 1.55fr, 1.18fr, 0.88fr, 0.72fr),
    align: (left, left, left, right, right),
    stroke: none,
    inset: (x: 4.5pt, y: 4.2pt),
    table.hline(stroke: 1.0pt),
    table.header(
      [*Satellite Method*], [*Forest Type*], [*Period*], [*Trend (/dec)*], [*vs. Base*],
    ),
    table.hline(stroke: 0.6pt),
    [Uncorrected Collection 2 baseline], [All stable forest], [Dec--Mar (1985--25)], [$+0.033$], [baseline],
    [Roy et al. (2016) Table 3 adjustment], [All stable forest], [Dec--Mar (1985--25)], [$+0.031$], [$-7\%$],
    [Overlap-calibrated (anchored to L7)], [All stable forest], [Dec--Mar (1985--25)], [$+0.009$], [$-72\%$],
    [*Overlap-calibrated (anchored to L7)*], [*Pure hardwood (NLCD 41)*], [*Jan--Feb (1985--25)*], [*$+0.007$*], [*$-79\%$*],
    [Single-sensor Landsat 5 only], [Pure hardwood (NLCD 41)], [Jan--Feb (1985--11)], [$+0.013$], [$-61\%$],
    [Single-sensor Landsat 5 only], [Pure evergreen (NLCD 42)], [Jan--Feb (1985--11)], [$+0.046$], [$+41\%$],
    [Single-sensor Landsat 5 only], [Regrown evergreen (85--01)], [Jan--Feb (1985--11)], [$+0.087$], [$+166\%$],
    [Single-sensor Landsat 7 only], [Pure hardwood (NLCD 41)], [Jan--Feb (2000--21)], [$+0.005$], [$-84\%$],
    [*Single-sensor Landsat 8+9 only*], [*Pure hardwood (NLCD 41)*], [*Jan--Feb (2014--25)*], [*$-0.008$*], [*Sign flip*],
    table.hline(stroke: 1.0pt),
  )
) <tab-optical>

Applying the Collection 1 adjustment of #cite(<Roy2016>, form: "prose") removes only 7% of the regional Collection 2 winter slope ($+0.033$ to $+0.031$/decade), whereas empirical overlap harmonization on pure hardwood forest reduces it by 79% (to $+0.007$/decade). Even the modest residual slope inside Landsat 5 alone ($+0.013$/decade during 1985--2011) is difficult to interpret as hardwood mesophication. Three diagnostics argue against that reading: (1) maturing 1985--2001 evergreen pine stands rose 6.5 times faster ($+0.087$/decade) than continuous old hardwoods ($+0.013$/decade), inflating unstratified averages; (2) Daymet V4 weather records show plateau winters cooled slightly during 1985--2011 (minimum temperature $-0.44$ #sym.degree;C/decade; snow days $+2.9$/decade), arguing against milder snow-free winters; and (3) over the same 3,197 km#super[2] of old hardwood forest, SWIR indices moved negatively during 1985--2011 ($"NBR" = -0.013$/decade; $"NDMI" = -0.016$/decade) and all three indices turned negative in Landsat 8+9 (2014--2025: $"NDVI" = -0.008$, $"NBR" = -0.043$, $"NDMI" = -0.052$). While these diagnostics rule out winter warming and conflict with SWIR moisture and biomass trends, the exact cause of the remaining $+0.013$/decade Landsat 5 NDVI slope (whether radiometric residuals or sub-pixel understory composition) cannot be determined from Landsat alone.

Calibration also alters the comparison across eight Tennessee protected areas (@tab-tn and @fig-sensor b). On pure upland hardwoods ($>= 430$ m), burned Sewanee Domain sloped slightly *higher* than unburned neighbor Franklin State Forest in both Landsat 5 ($+0.012$ vs. $+0.010$) and Landsat 7 ($+0.007$ vs. $+0.001$). Seven of the eight protected areas, including the repaired 10,282 ha Fall Creek Falls State Park, sloped slightly negative during 2014--2025. Annual NLCD maps confirm that plateau-top forest area stepped down from 5,617 km#super[2] in 1985 to 5,135 km#super[2] in 2001 ($-8.6\%$) and remained stable at 5,185 km#super[2] in 2024 ($+1.0\%$ net despite 1,129 km#super[2] of rotational harvest regrowth).

#figure(
  kind: table,
  caption: [*Calibrated Landsat winter NDVI trends (per decade) across eight Tennessee protected areas on the upland plateau ($>= 430$ m elevation, Annual NLCD composition, repaired 10,282 ha Fall Creek Falls boundary).*],
  table(
    columns: (1.88fr, 0.68fr, 0.62fr, 0.84fr, 0.64fr, 0.82fr, 0.80fr, 0.90fr),
    align: (left, right, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 3.2pt, y: 4pt),
    table.hline(stroke: 1.0pt),
    table.header(
      [*Protected Area*], [*Area (ha)*], [*$>=430$ m*], [*Hardw./Pine*], [*Uncorr.*], [*Calib. Hardw.*], [*L5 (85--11)*], [*L8+9 (14--25)*],
    ),
    table.hline(stroke: 0.6pt),
    [Prentice Cooper SF], [10,009], [75.4%], [93.4 / 0.5%], [$+0.025$], [$-0.002$], [$+0.004$], [$-0.021$],
    [N. Chickamauga Gorge], [2,885], [80.7%], [61.3 / 5.9%], [$+0.032$], [$+0.002$], [$+0.008$], [$-0.008$],
    [Sewanee Domain (LAL)], [5,190], [83.4%], [97.8 / 0.1%], [$+0.030$], [$+0.003$], [$+0.012$], [$-0.008$],
    [Franklin State Forest], [3,564], [83.2%], [95.1 / 0.8%], [$+0.030$], [$+0.004$], [$+0.010$], [$-0.004$],
    [Catoosa WMA], [32,048], [82.9%], [76.1 / 0.6%], [$+0.032$], [$+0.008$], [$+0.005$], [$+0.001$],
    [Savage Gulf SNA], [6,896], [81.3%], [70.8 / 4.6%], [$+0.035$], [$+0.008$], [$+0.015$], [$-0.012$],
    [Fall Creek Falls SP (fixed)], [10,282], [81.5%], [69.7 / 7.3%], [$+0.038$], [$+0.010$], [$+0.021$], [$-0.007$],
    [Bridgestone/Firestone WMA], [4,915], [68.8%], [77.1 / 5.3%], [$+0.037$], [$+0.014$], [$+0.022$], [$-0.010$],
    table.hline(stroke: 1.0pt),
  )
) <tab-tn>

== Phase 2: Multi-Epoch Airborne LiDAR BACI: Identified Fire Contrasts and Honest Bounds

Before interpreting multi-epoch airborne LiDAR shifts, flight-hardware diagnostics establish an essential identification boundary. Between the baseline and follow-up acquisitions, both mean return density and multi-return sensitivity increased substantially: in Bankhead North (2012 QL3 to 2021 QL2), mean point density rose from 3.5 to 14.7 pts/m#super[2] while the fraction of first returns fell from 0.43 to 0.32; even in Daniel Boone East's QL2-to-QL2 pair (2017 to 2023), mean point density rose from 6.5 to 11.3 pts/m#super[2] and the first-return fraction fell from 0.59 to 0.44. Random return thinning equalizes point density, not sensor hardware, scan geometry, or multi-return recording depth. Consequently, the positive raw shifts observed in unburned controls ($+0.113$ in Bankhead; $+0.034$ in Daniel Boone) cannot be interpreted on their own as pure ecological densification. Because burned and unburned plots inside each forest share the exact same baseline and follow-up flights, *only the burned-minus-unburned Difference-in-Differences (DiD) is causally identified* (@fig-baci and @tab-baci).

#figure(
  image("../figures/fig2_baci_lidar_structural_change.png", width: 98%),
  caption: [*Repeat-LiDAR BACI midstory change (2--10 m NRD) in Alabama and Kentucky.* Bars show plot-level means with 95% bootstrap CIs and density-thinned checks (white diamonds) for (a) Bankhead NF (2012--2021, $n = 139$) and (b) Daniel Boone NF (2017--2023, $n = 144$). Because inter-flight sensor and multi-return changes contribute to raw bar heights, only the DiD relative to unburned controls is identified. Annotations report burn-unit clustered $p_u$ ($p_"SUID"$); sample labels list both plot counts ($n$) and unique burn units ($G$).],
) <fig-baci>

#figure(
  kind: table,
  caption: [*Multi-epoch airborne LiDAR BACI comparisons (2--10 m midstory NRD) alongside baseline/endpoint levels, density-thinned DiD, baseline-adjusted ANCOVA DiD, and Landsat winter/spring optical DiD.*],
  table(
    columns: (1.58fr, 0.50fr, 0.86fr, 0.50fr, 1.56fr, 1.02fr, 0.96fr),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 2.2pt, y: 3.5pt),
    table.hline(stroke: 1.0pt),
    table.header(
      [*Forest & Stratum*], [*$n$ ($G$)*], [*Base #sym.arrow End*], [*Raw $Delta$*], [*DiD ($p_"SUID"$ / $p_"WCB"$)*], [*Thin / ANC*], [*Win / Spr $Delta$NDVI*],
    ),
    table.hline(stroke: 0.6pt),
    table.cell(colspan: 7)[*Bankhead North, AL (2012--2021, 8.9 yr; first-return share 0.43 #sym.arrow 0.32)*],
    [Unburned control (0 burns)], [63 (45)], [0.280 #sym.arrow 0.393], [$+0.113$], [ref], [ref], [ref],
    [1 burn (2012--20)], [34 (14)], [0.238 #sym.arrow 0.285], [$+0.047$], [*$-0.066$ ($<$.001 / $<$.001)*], [$-0.067$ / $-0.071$], [$+0.002$ / $-0.045$],
    [2+ burns (2012--20)], [36 (17)], [0.219 #sym.arrow 0.255], [$+0.037$], [*$-0.076$ ($<$.001 / $<$.001)*], [$-0.078$ / $-0.083$], [$-0.005$ / $+0.001$],
    table.hline(stroke: 0.4pt),
    table.cell(colspan: 7)[*Daniel Boone East, KY (2017--2023, 6.0 yr; uniform 2017 baseline; first-return share 0.59 #sym.arrow 0.44)*],
    [Unburned control (0 burns)], [61 (59)], [0.191 #sym.arrow 0.225], [$+0.034$], [ref], [ref], [ref],
    [Pre-2017 burn only (0 inter)], [20 (3)], [0.127 #sym.arrow 0.188], [$+0.061$], [$+0.027$ (.090 / .112)], [$+0.028$ / $-0.001$], [$-0.014$ / $+0.004$],
    [1 burn (2018--22)], [28 (4)], [0.148 #sym.arrow 0.190], [$+0.042$], [$+0.009$ (.754 / .713)], [$+0.009$ / $-0.010$], [$-0.014$ / $-0.010$],
    [Pre-2017 mech + 1 burn], [35 (4)], [0.125 #sym.arrow 0.173], [$+0.049$], [$+0.015$ (.096 / .147)], [$+0.016$ / $-0.014$], [$-0.003$ / $+0.005$],
    table.hline(stroke: 1.0pt),
  )
) <tab-baci>

In *Bankhead North*, prescribed fire produced large, tightly identified structural reductions relative to unburned controls. One inter-epoch burn reduced 2--10 m midstory NRD change by $-0.066$ ($d = -0.82$, Welch $p < 0.001$, burn-unit $p_"SUID" < 0.001$, wild cluster bootstrap $p_"WCB" = 0.0001$, 2 km block $p < 0.001$). Two or more inter-epoch burns reduced midstory change by $-0.076$ ($d = -0.95$, Welch $p < 0.001$, $p_"SUID" < 0.001$, $p_"WCB" = 0.0001$), alongside a $-0.041$ reduction in 0.5--2 m understory change ($+0.016$ vs. $+0.057$). Both estimates are invariant to exact density thinning ($-0.067$ and $-0.078$) and strengthen slightly under baseline-adjusted ANCOVA ($-0.071$ and $-0.083$, both $p_"SUID" < 0.001$). Unbiased first-return upper canopy cover ($>10$ m) showed a raw negative DiD under inter-epoch burning ($-0.049, p_"SUID" = 0.041$ for 1 burn; $-0.045, p_"SUID" = 0.004$ for 2+ burns). Baseline and endpoint decomposition clarifies why this does not contradict the null cross-sectional overstory result: burned stands started 4--6 percentage points *higher* than controls in 2012 ($0.725$ and $0.712$ vs. $0.669$) and ended statistically indistinguishable from controls in 2021 ($0.745$ and $0.736$ vs. $0.741$, $p_"SUID" = 0.88$ and $0.78$). Under baseline-adjusted ANCOVA, the 1-burn effect turns null ($-0.027, p_"SUID" = 0.18$) while the 2+ burn contrast halves ($-0.028, p_"SUID" = 0.012$), with unchanged 95th-percentile treetop height (`CHM p95`: $+1.27$ m vs. $+1.16$ m).

In *Daniel Boone East*, exact-date auditing and small-cluster bootstrap checks reveal three important nuances that guard against over-interpretation. First, a single burn between 2017 and 2023 ($n = 28, G = 4$ units) produced no detectable midstory reduction relative to controls ($"DiD" = +0.009, p_"SUID" = 0.75, p_"WCB" = 0.71$). Second, plots burned only prior to 2017 ($n = 20$, concentrated in just $G = 3$ burn units with 16 of 20 plots in a single unit) showed a positive raw DiD ($+0.027$, Welch $p = 0.033$) that is *not* statistically significant under burn-unit clustering ($p_"SUID" = 0.090$; wild cluster bootstrap $p_"WCB" = 0.112$; 2 km blocks $p = 0.063$). More importantly, those plots began at a significantly lower 2017 baseline than controls ($0.127$ vs. $0.191$, Welch $p = 0.007$), remained *below* controls in 2023 ($0.188$ vs. $0.225$, $p = 0.060$; understory $0.046$ vs. $0.045, p = 0.91$), and showed zero differential growth once baseline midstory density is controlled in ANCOVA ($-0.001, p_"SUID" = 0.94$). This pattern reflects expected post-fire recovery from a suppressed baseline rather than a rebound overshooting unburned stands. Third, auditing FACTS mechanical-treatment dates shows that *all 35* plots in Daniel Boone's `baci_mech_fire` stratum were mechanically thinned *before* the 2017 baseline (FY 2008--2013; calendar dates 2007--2013; zero inter-epoch mechanical plots) and received only one prescribed burn during 2018--2022. That explains both their low 2017 starting density ($0.125$ vs. $0.191$), their persistently lower 2023 endpoint ($0.173$ vs. $0.225$, $p < 0.001$), and their null 2017--2023 DiD ($+0.015, p_"SUID" = 0.096, p_"WCB" = 0.147$; ANCOVA $-0.014, p_"SUID" = 0.10$). Because Bankhead North had only two inter-epoch mechanical plots (within $n = 4, G = 3$), our repeat-LiDAR dataset does *not* provide a before/after test of inter-epoch mechanical thinning.

== Cross-Sectional Vertical Structure: Continuous 1 m Profile, Direct Contrasts, and Unbiased Canopy

Across all 689 plots, pooled site-fixed-effects regressions with burn-unit clustering ($G = 359$) reveal how surface fire and mechanical thinning + fire differ across vertical height, exposing two critical findings missed by coarse strata or all-return canopy metrics (@fig-decoupling and @tab-feols).

#figure(
  image("../figures/fig3_vertical_stratum_decoupling.png", width: 99%),
  caption: [*Continuous 1-meter vertical response profile and unbiased first-return canopy cover ($n = 689$).* (a) Pooled site-FE OLS coefficients across 1 m height bins (0.5--10 m, 95% burn-unit CIs) show that `3+` burns attenuate gradually with height and cross over `mech+fire` across a 4--7 m transition zone; asterisks mark bins where the direct cluster-robust `mech - 3+` contrast has $p_u < 0.05$ (`*`) or $< 0.01$ (`**`). (b) Aggregated stratum effects across all four treatment categories using unbiased first-return cover above 10 m.],
) <fig-decoupling>

#figure(
  kind: table,
  caption: [*Pooled multi-forest fixed-effects OLS across 3DEP LiDAR vertical metrics and Landsat seasons ($n = 689$; winter Landsat $n = 678$). Parentheses give burn-unit clustered $p_"SUID"$ ($G = 359$); rightmost column tests the direct linear contrast `mech+fire` minus `3+` burns.*],
  table(
    columns: (1.82fr, 0.34fr, 0.97fr, 0.97fr, 0.97fr, 0.97fr, 0.97fr),
    align: (left, right, right, right, right, right, right),
    stroke: none,
    inset: (x: 2.4pt, y: 3.5pt),
    table.hline(stroke: 1.0pt),
    table.header(
      [*Metric or Sensor*], [*$R^2$*], [*1 Burn*], [*2 Burns*], [*3+ Burns*], [*Mech + Fire*], [*Mech #sym.minus 3+*],
    ),
    table.hline(stroke: 0.6pt),
    [LiDAR Sprouts (0.5--2 m)], [0.19], [$-0.013$ (.002)], [$-0.021$ ($<$.001)], [*$-0.027$ ($<$.001)*], [$-0.010$ (.069)], [$+0.017$ (.002)],
    [LiDAR Low Mid. (2--5 m)], [0.41], [$-0.032$ ($<$.001)], [$-0.045$ ($<$.001)], [*$-0.060$ ($<$.001)*], [$-0.034$ ($<$.001)], [$+0.025$ (.014)],
    [LiDAR Tall Mid. (5--10 m)], [0.28], [$-0.023$ (.007)], [$-0.014$ (.122)], [$-0.017$ (.164)], [*$-0.042$ ($<$.001)*], [$-0.024$ (.106)],
    [LiDAR Midstory (2--10 m)], [0.36], [$-0.046$ ($<$.001)], [$-0.050$ ($<$.001)], [*$-0.064$ ($<$.001)*], [*$-0.067$ ($<$.001)*], [$-0.003$ (.887)],
    [*1st-Return Canopy ($>10$ m)*], [0.41], [$-0.015$ (.362)], [$+0.022$ (.193)], [$+0.032$ (.117)], [*$-0.067$ (.001)*], [*$-0.098$ ($<$.001)*],
    [All-Return Share ($>10$ m)#super[a]], [0.39], [$+0.011$ (.316)], [$+0.029$ (.002)], [$+0.037$ (.002)], [$-0.024$ (.056)], [$-0.061$ ($<$.001)],
    [Canopy Height (`CHM p95`, m)], [0.25], [$+0.40$ (.384)], [$+0.44$ (.256)], [$-0.20$ (.721)], [$+0.87$ (.233)], [$+1.07$ (.227)],
    [Vertical Diversity (FHD)], [0.54], [$-0.048$ (.167)], [$-0.022$ (.436)], [$-0.032$ (.299)], [*$-0.154$ ($<$.001)*], [$-0.122$ (.006)],
    [Landsat Winter NDVI ($n=678$)], [0.19], [$-0.007$ (.283)], [$-0.003$ (.632)], [$+0.005$ (.352)], [$-0.004$ (.564)], [$-0.010$ (.204)],
    [Landsat Spring NDVI (Mar--Apr)], [0.44], [*$-0.028$ ($<$.001)*], [$-0.016$ (.065)], [*$-0.019$ (.006)*], [*$-0.021$ (.002)*], [$-0.002$ (.819)],
    table.hline(stroke: 1.0pt),
  )
) <tab-feols>
#text(size: 8.2pt, fill: rgb("#444444"))[#super[a]Retained solely to illustrate mechanical upward bias when subcanopy returns are removed from $N_"total"$; unbiased inference uses first-return canopy cover above it.]

First, examining the *continuous 1-meter vertical profile* (@fig-decoupling a) shows that surface-fire effects do not hit a sharp cliff at an arbitrary 5 m bin edge; instead, they attenuate smoothly with height. Under `3+` surface burns, NRD reductions peak at 2--3 m ($-0.029, p_"SUID" < 0.001$), remain strong at 3--4 m ($-0.022$) and 4--5 m ($-0.014, p_"SUID" < 0.001$), stay statistically significant at 5--6 m ($-0.009, p_"SUID" = 0.020$), and fade to statistical null above 6 m (6--7 m: $-0.006, p_"SUID" = 0.075$; 7--10 m: $-0.003$ to $-0.001, p_"SUID" = 0.29$--$0.80$). By contrast, cross-sectional `mech+fire` stands exhibit a nearly uniform $-0.008$ to $-0.013$ reduction in every 1 m slice from 2 m through 10 m ($p_"SUID" < 0.001$). Testing the direct cluster-robust contrast `mech - 3+` clarifies where the two regimes genuinely diverge: `3+` burns significantly outperform `mech+fire` in the 1--4 m zone ($p_"SUID" <= 0.011$), the two regimes overlap indistinguishably across a 4--7 m crossover zone ($p_"SUID" = 0.28$--$0.70$), and `mech+fire` significantly outperforms `3+` burns only in the upper 8--10 m midstory (8--9 m: $-0.007, p_"SUID" = 0.024$; 9--10 m: $-0.008, p_"SUID" = 0.006$). In the aggregated 5--10 m bin, the direct `mech - 3+` difference ($-0.024$) does not reach significance at $alpha = 0.05$ ($p_"SUID" = 0.106$), and fire-only effects are non-monotone across burn counts (`1 burn`: $-0.023, p_"SUID" = 0.007$; `2 burns`: $-0.014, p_"SUID" = 0.12$; `3+ burns`: $-0.017, p_"SUID" = 0.16$), likely reflecting site-selection differences in which stands receive repeated burns.

Second, replacing the mechanically biased all-return canopy share with *unbiased first-return canopy cover ($>10$ m)* changes the overstory conclusion sharply (@tab-feols and @fig-decoupling b). Under the biased all-return ratio, `3+` burns appeared to increase $>10$ m canopy ($+0.037, p_"SUID" = 0.002$) while `mech+fire` looked borderline ($-0.024, p_"SUID" = 0.056$). Under unbiased first-return cover, `1`, `2`, and `3+` surface burns show *no* statistically significant overstory difference ($-0.015, +0.022, +0.032$; $p_"SUID" = 0.12$--$0.36$, consistent with unchanged `CHM p95`), whereas `mech+fire` shows a substantial 6.7 percentage point *reduction* in upper canopy cover ($-0.067, p_"SUID" = 0.001$; direct `mech - 3+` contrast $-0.098, p_"SUID" < 0.001$), alongside lower Foliage Height Diversity ($-0.154, p_"SUID" < 0.001$). Mechanically removing stems naturally reduces upper-midstory and subcanopy crown packing by design, and in these USFS FACTS stands it also opens the $>10$ m canopy noticeably. Dropping the 9 cross-site plots within 220 m of a neighboring site ($n = 680$) leaves every coefficient virtually unchanged (`3+` midstory $-0.064, p_"SUID" < 0.001$; `mech+fire` 5--10 m $-0.042, p_"SUID" < 0.001$).

Topographically matched comparisons within individual forests confirm consistent cross-sectional reductions under repeated fire in Alabama and mixed patterns in Kentucky: pooled across sites, `3+` burns hold 29.1% less 2--10 m midstory than matched controls ($-0.088, d = -0.79, p < 0.001$), reaching 32.4% in Bankhead North ($-0.119, d = -1.13$) and 48.1% in Bankhead South ($-0.159, d = -1.95$, matched with replacement because 35 treated plots exceed 31 controls). In Daniel Boone Main (2017), `3+` burns showed sharply lower understory sprouts (0.017 vs. 0.046) and low midstory (0.062 vs. 0.088) yet higher 5--10 m density (0.180 vs. 0.122), whereas `mech+fire` stands held 27.8% lower overall 2--10 m midstory than matched controls ($d = -0.65, p = 0.006$).

== Head-to-Head Sensor Comparison: Why Neither Winter Nor Spring Landsat Tracks Temporal Change

Evaluating Landsat seasons and NASA GEDI L2B directly against 3DEP LiDAR on identical plots shows that *passive optical satellites fail the temporal change-detection test in both winter and spring* (@fig-sensor a):

#figure(
  image("../figures/fig4_sensor_comparison_and_tn_reversal.png", width: 99%),
  caption: [*Pooled versus within-site sensor correlations, BACI delta failure, and Tennessee protected-area trend sensitivity.* (a) Cross-sectional correlations with 3DEP LiDAR 2--10 m midstory shrink and converge once between-site differences are removed (solid bars: within-site $r$; pale bars: pooled $r$), and temporal BACI optical deltas in both winter and spring cluster near zero. (b) Shift in plateau-top ($>= 430$ m) winter NDVI slopes across eight Tennessee protected areas from uncorrected archives to overlap-calibrated hardwood and 2014--2025 Landsat 8+9 eras.],
) <fig-sensor>

1. *Within-site cross-sectional skill is modest and nearly identical across sensors:* Pooled across all five sites, March--April spring NDVI shows a higher cross-sectional correlation with 2--10 m LiDAR midstory ($r = +0.41, n = 689$) than January--February winter NDVI ($r = +0.33, n = 678$) or GEDI 5--10 m PAI ($r = +0.14, n = 359$). Removing site means reveals that most of spring's apparent lead is between-site: *within sites*, spring NDVI ($r = +0.30$), winter NDVI ($r = +0.28$), and GEDI 5--10 m PAI ($r = +0.29$) perform almost identically. In fact, winter NDVI correlates more strongly than spring NDVI inside Daniel Boone Main ($r = +0.52$ vs. $+0.14$) and Sewanee/Franklin ($r = +0.55$ vs. $+0.48$), whereas spring leads in Bankhead North ($+0.37$ vs. $+0.12$), Bankhead South ($+0.33$ vs. $+0.20$), and Daniel Boone East ($+0.50$ vs. $+0.21$). Summer NDVI shows essentially no link ($r = +0.01$ pooled, $+0.05$ within-site).
2. *Cross-sectional burn sensitivity differs between seasons, but temporal BACI change fails in both:* In pooled site-FE regressions (@tab-feols), winter NDVI detects no fire effect at any burn count ($p_"SUID" = 0.28$--$0.63$), whereas spring NDVI does respond negatively to burning (`1 burn`: $-0.028, p_"SUID" < 0.001$; `3+ burns`: $-0.019, p_"SUID" = 0.006$; `mech+fire`: $-0.021, p_"SUID" = 0.002$), consistent with early-leafing understory shrubs and saplings in unburned stands. However, when we apply the *same temporal BACI standard* on which winter NDVI fails, spring NDVI fails as well. Across epoch-verified BACI plots, 2014--2016 to 2020--2023 winter $Delta$NDVI shows zero correlation with $Delta$NRD midstory ($r = -0.001, p = 0.98, n = 272$; within-site $r = +0.009, p = 0.89$; Bankhead $r = +0.20, p = 0.016$; Daniel Boone $r = -0.05, p = 0.58$), and *spring $Delta$NDVI likewise fails* ($r = -0.054, p = 0.37, n = 283$; within-site $r = +0.076, p = 0.20$; Bankhead $r = +0.12, p = 0.15$; Daniel Boone $r = -0.09, p = 0.29$). Aligning the spring window to Daniel Boone's exact 2017--2023 flights (2016--2018 vs. 2022--2024) yields $r = -0.020$ pooled ($p = 0.74$) and $r = -0.004$ in Daniel Boone ($p = 0.96$). Stratum-level spring DiDs also swing inconsistently in sign. In Bankhead North, `1 burn` showed a negative spring DiD ($-0.045, p_"SUID" = 0.006$), yet `2+` burns (which produced our largest 3D LiDAR reduction, $-0.076, p_"SUID" < 0.001$) showed zero spring NDVI shift ($+0.001, p_"SUID" = 0.96$). In Daniel Boone's epoch-aligned window (2016--2018 to 2022--2024), `1 burn` moved *positively* ($+0.046, p_"SUID" < 0.001$; `pre-2017 mech + 1 burn` $+0.038, p_"SUID" = 0.016$) despite null LiDAR midstory change. Year-to-year phenological timing shifts and post-burn herbaceous flush in March--April easily swamp multi-year woody structural signals.
3. *Spaceborne NASA GEDI is informative within sites but spatially sparse:* Across 359 plots with footprints within 75 m, GEDI 5--10 m PAI reaches $r = +0.29$ within-site ($p < 0.001$), matching optical snapshots, while GEDI total PAI ($+0.15$) and FHD ($+0.04$) remain weak. Sparse (roughly 60 m) along-track sampling across mixed leaf seasons limits GEDI to regional summaries rather than individual burn-unit monitoring.

= Synthesis, Interactive Explorer, and Limitations

== What the Evidence Supports and What Remains Untested

To make these data transparent and inspectable for foresters and public readers alike, we packaged all 689 plots, the calibrated Tennessee baseline, and the updated statistical outputs into a standalone interactive web explorer (`explorer.html`, using keyless USGS/Esri basemaps and an embedded Google Earth Engine forest-and-relief layer). Box 2 summarizes what our measurements support directly alongside explicit boundaries where causal evidence is still lacking.

#v(2pt)
#callout_box("Box 2. Summary of Empirical Support and Monitoring Implications")[
  #table(
    columns: (1.25fr, 1.65fr, 1.45fr),
    align: (left, left, left),
    stroke: none,
    inset: (x: 4pt, y: 3.5pt),
    table.hline(stroke: 0.8pt),
    table.header([*Question*], [*What Our Data Support*], [*Key Caveat / Monitoring Note*]),
    table.hline(stroke: 0.4pt),
    [Does prescribed fire reduce subcanopy crowding?], [*Yes, strongly in Bankhead BACI* (`1` burn $"DiD" = -0.066$, `2+` burns $-0.076$, $p_"SUID" < 0.001$) and cross-sectionally across sites below roughly 5--6 m.], [Only burned-minus-unburned DiD is identified; raw control shifts include inter-flight sensor changes. A single burn in Daniel Boone BACI showed null 6-year DiD ($+0.009$).],
    [How do surface fire and mechanical + fire differ vertically?], [In 1 m cross-sectional profiles, `3+` burns beat `mech+fire` at 1--4 m ($p_"SUID" <= 0.011$) while leaving first-return $>10$ m canopy unchanged; `mech+fire` is stronger at 8--10 m ($p_"SUID" <= 0.024$) and opens $>10$ m canopy ($-0.067$).], [Cross-sectional only: all Daniel Boone BACI mechanical thinning predated 2017. Lower pole density after physical cutting is expected by design; direct 5--10 m contrast is n.s. ($p_"SUID" = 0.11$).],
    [What happens when burning stops?], [Pre-2017 burned stands in Daniel Boone recovered from a lower starting baseline ($0.127$ #sym.arrow $0.188$) without exceeding unburned controls ($0.225$).], [Concentrated in just 3 burn units ($p_"SUID" = 0.090$, $p_"WCB" = 0.11$; ANCOVA $-0.001, p = 0.94$). Insufficient to claim overshoot rebound.],
    [Can Landsat replace repeat airborne LiDAR?], [*No.* Neither winter ($r = -0.00$) nor spring ($r = -0.05$) Landsat NDVI tracks multi-year BACI structural change. Use repeat leaf-off 3DEP LiDAR with matched controls.], [Spring NDVI detects burn regime cross-sectionally ($p_"SUID" <= 0.006$), but within-site $r = +0.30$ barely differs from winter ($+0.28$) and fails temporal change validation.],
    table.hline(stroke: 0.8pt),
  )
]

== Limitations and Next Steps

Five clear limitations bound this study. First, USFS FACTS polygons record mapped burn boundaries rather than continuous fire intensity or unburned damp micro-refugia inside a unit, which attenuates measured treatment contrasts toward zero. Second, because repeat 3DEP flights differ in sensor hardware, scan angle, and multi-return depth (`frac_first_returns` fell from 0.43 to 0.32 in Bankhead and from 0.59 to 0.44 in Daniel Boone), random thinning checks density sensitivity but does not identify absolute unburned growth rates; our causal inferences therefore rest strictly on within-flight DiD contrasts. Third, our pure-deciduous filter intentionally excludes mixed shortleaf-pine savanna restorations (such as portions of Catoosa WMA), which require conifer-stratified analysis. Fourth, leaf-off airborne LiDAR quantifies woody stem packing without identifying species; pairing these 689 plots with U.S. Forest Service FIA inventory plots or Sewanee permanent compartments is needed to separate red maple and beech from oak regeneration directly. Finally, Tennessee's 2015--2016 statewide 3DEP scan predates 95% of Sewanee's mapped burn acreage, and our repeat-LiDAR flights contain too few inter-epoch mechanical treatments to evaluate mechanical + fire longitudinally; future repeat 3DEP acquisitions across Tennessee and recently thinned national-forest units will allow direct BACI tests of both.
