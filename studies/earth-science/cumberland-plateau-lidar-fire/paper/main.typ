#set document(
  title: "Seeing Beneath the Canopy: Multi-Epoch Airborne LiDAR Evaluates Leaf-Off Landsat Proxies and Vertical Prescribed-Fire Effects on the Cumberland Plateau",
  author: ("Daksh Tyagi", "Gemini 4 Neon"),
)

#set page(
  paper: "us-letter",
  margin: (top: 0.85in, bottom: 0.85in, left: 0.90in, right: 0.90in),
  header: context {
    if counter(page).get().first() > 1 [
      #set text(size: 8.5pt, fill: rgb("#555555"))
      #grid(
        columns: (1fr, auto),
        align: (left, right),
        smallcaps[Seeing Beneath the Canopy: Airborne LiDAR & Prescribed Fire],
        counter(page).display(),
      )
      #v(-4pt)
      #line(length: 100%, stroke: 0.4pt + rgb("#cccccc"))
    ]
  },
  footer: context {
    if counter(page).get().first() == 1 [
      #align(center)[#text(size: 8.5pt, fill: rgb("#555555"))[1]]
    ]
  },
)

#set text(
  font: "New Computer Modern",
  size: 9.9pt,
  fill: rgb("#111111"),
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.60em,
  spacing: 0.88em,
  first-line-indent: (amount: 1.4em, all: false),
)

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  v(13pt, weak: true)
  block(sticky: true)[
    #set text(size: 11.8pt, weight: "bold", fill: rgb("#111111"))
    #if it.numbering != none {
      counter(heading).display()
      h(0.55em)
    }
    #it.body
  ]
  v(5pt, weak: true)
}
#show heading.where(level: 2): it => {
  v(9pt, weak: true)
  block(sticky: true)[
    #set text(size: 10.4pt, weight: "bold", fill: rgb("#222222"))
    #if it.numbering != none {
      counter(heading).display()
      h(0.45em)
    }
    #it.body
  ]
  v(4pt, weak: true)
}

#show figure.caption: it => {
  set text(size: 8.7pt, fill: rgb("#262626"))
  set par(leading: 0.52em, justify: true, first-line-indent: 0pt)
  block(width: 98%)[#it]
}
#show figure.where(kind: table): set figure.caption(position: top)
#set figure(gap: 7pt)
#show figure: it => {
  v(7pt, weak: true)
  it
  v(7pt, weak: true)
}

#show raw.where(block: false): it => text(font: "New Computer Modern Mono", size: 8.8pt, fill: rgb("#1c3a30"))[#it]

#show table: set text(size: 7.8pt)
#show table: set par(justify: false, leading: 0.45em, first-line-indent: 0pt)

#align(center)[
  #block(width: 94%)[
    #set par(leading: 0.55em, justify: false)
    #text(size: 13.8pt, weight: "bold")[
      Seeing Beneath the Canopy: Multi-Epoch Airborne LiDAR\
      Evaluates Leaf-Off Landsat Proxies and Vertical\
      Prescribed-Fire Effects on the Cumberland Plateau
    ]
  ]
  #v(10pt)
  #text(size: 10.8pt)[#smallcaps[Daksh Tyagi]]
  #v(3pt)
  #text(size: 10.5pt)[#smallcaps[Gemini 4 Neon]]
  #v(5pt)
  #text(size: 9.8pt, fill: rgb("#444444"))[November 2026]
]

#v(11pt)

#include "sections/abstract.typ"

#v(9pt)

#block(
  width: 100%,
  fill: rgb("#f6f8f7"),
  stroke: (top: 0.9pt + rgb("#1d4d3e"), bottom: 0.7pt + rgb("#c4cfc9")),
  inset: (x: 11pt, y: 8pt),
  breakable: false,
)[
  #text(weight: "bold", size: 9.2pt, fill: rgb("#163b30"))[#smallcaps[Key Takeaways]]
  #v(3pt)
  #set text(size: 8.9pt)
  #set par(leading: 0.54em, justify: false, first-line-indent: 0pt)
  - *Landsat NDVI misses temporal deciduous midstory change:* Removing the Collection 2 inter-sensor step cuts the apparent 40-year winter trend by *79%*, and against repeat airborne LiDAR, *neither* winter ($r = -0.00$) *nor* spring ($r = -0.05$) NDVI tracks multi-year subcanopy change.
  - *Only BACI Difference-in-Differences (DiD) is identified:* Because newer 3DEP flights record more multiple returns per pulse, raw control shifts mix growth with sensor change; causal inference rests on burned-minus-unburned DiD.
  - *Strong Bankhead BACI suppression; baseline recovery in Kentucky:* In Bankhead NF (2012--2021), 1 and 2+ burns reduced 2--10 m midstory change by *$-0.066$* and *$-0.076$* NRD ($p_"SUID" < 0.001$, $p_"WCB" = 0.0001$), invariant to density thinning; in Daniel Boone NF (2017--2023), a single burn had null DiD ($+0.009$) and pre-2017 burns (3 units) recovered toward, not past, controls.
  - *Gradual vertical attenuation and unbiased first-return canopy:* Across 1 m height bins ($n = 689$), `3+` burns beat `mech+fire` at 1--4 m ($p_"SUID" <= 0.011$), cross over at 4--7 m, and leave unbiased first-return $>10$ m canopy unchanged; cross-sectional `mech+fire` is stronger at 8--10 m ($p_"SUID" <= 0.024$) and opens $>10$ m canopy ($-0.067, p_"SUID" = 0.001$).
  - *Within-site optical correlations converge:* Spring NDVI ($r = +0.30$), winter NDVI ($+0.28$), and NASA GEDI 5--10 m PAI ($+0.29$) perform similarly within sites, confirming repeat airborne LiDAR cannot be replaced by passive optical proxies.
]

#v(6pt)

#include "sections/introduction.typ"
#include "sections/body.typ"
#include "sections/conclusion.typ"

#v(4pt)
#[
  #set text(size: 9.2pt)
  #bibliography("references.bib", style: "elsevier-harvard", title: "References")
]
