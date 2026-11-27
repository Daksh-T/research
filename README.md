# research

Open, reproducible research studies directed by **Daksh Tyagi** and executed with agent swarms. These are published alongside full analysis pipelines, manuscripts, and adversarial peer-review (by another independant model) records.

Each directory under [`studies/`](studies/) is a self-contained research package (single study or multi-paper family) with its own `README.md`, machine-readable `study.yaml`, manuscript PDF and source (LaTeX or Typst), analysis code, and referee audit trail.

---

## Studies

| Discipline | Study | Papers | Artifacts |
|---|---|---:|---|
| **Earth Science** | [`cumberland-escarpment-phenology`](studies/earth-science/cumberland-escarpment-phenology/) | 1 | [PDF](studies/earth-science/cumberland-escarpment-phenology/paper/main.pdf) · [LaTeX](studies/earth-science/cumberland-escarpment-phenology/paper/main.tex) · [Notebook](studies/earth-science/cumberland-escarpment-phenology/Escarpment_Hopkins_Karst_v2.ipynb) · [Review](studies/earth-science/cumberland-escarpment-phenology/review/peer_review.md) |
| **Earth Science** | [`cumberland-plateau-lidar-fire`](studies/earth-science/cumberland-plateau-lidar-fire/) | 1 | [PDF](studies/earth-science/cumberland-plateau-lidar-fire/paper/cumberland_plateau_lidar_fire.pdf) · [Typst](studies/earth-science/cumberland-plateau-lidar-fire/paper/main.typ) · [Explorer](studies/earth-science/cumberland-plateau-lidar-fire/explorer.html) · [Review](studies/earth-science/cumberland-plateau-lidar-fire/review/response_to_referee.md) |
| **Economics** | [`us-immigrant-integration`](studies/economics/us-immigrant-integration/) | 3 | [Paper 1 (Intermarriage)](studies/economics/us-immigrant-integration/papers/paper2_intermarriage_comment/paper2_manuscript.pdf) · [Paper 2 (GenAI)](studies/economics/us-immigrant-integration/papers/paper1_genai_immigrants/paper1_manuscript.pdf) · [Paper 3 (Enclaves)](studies/economics/us-immigrant-integration/papers/paper3_enclaves_wfh/paper3_manuscript.pdf) · [Reviews](studies/economics/us-immigrant-integration/review/response_to_referee.md) |

See [`CONTENTS.md`](CONTENTS.md) for the complete paper-by-paper catalog, abstracts, and data-acquisition summary.

---

## Methodology, Provenance & Epistemic Standards

This repository is an experiment in open-science for responsible, human-directed agentic research:

- **Human Direction & Accountability:** Daksh Tyagi selects research questions, supervises as the agents come up with designs and audits, and bears responsibility for corrections and errata.
- **Agent Execution:** Pipelines, econometric/geospatial runs, visualizations, and manuscript drafts are executed by agent swarms directly against public-use data, with the specific models and tooling documented in each study's provenance metadata.
- **Independent Adversarial Review:** Before release, manuscripts and underlying code undergo adversarial code-level review by other independent frontier model(s). Full referee reports and point-by-point responses are published under each study's `review/` folder, including null findings and retracted preliminary claims.
- **Working-Paper Status:** These manuscripts are open working papers and replication/extension studies. They are not peer-reviewed journal articles and must not be considered published literature. Corrections, replications, and errata are welcome via GitHub Issues and logged transparently in [`CHANGELOG.md`](CHANGELOG.md).

---

## Repository Structure

```text
research/
├── README.md                  # Repository overview and index
├── CONTENTS.md                # Full paper-by-paper catalog across disciplines
├── CHANGELOG.md               # Versioned releases and errata log
├── CITATION.cff               # Machine-readable repository citation
├── LICENSE                    # Dual license summary (MIT code / CC-BY-4.0 content)
├── templates/study-template/  # Boilerplate for adding a new study or discipline
└── studies/
    └── <discipline>/
        └── <study-slug>/
            ├── README.md      # Study-specific findings and reproduction steps
            ├── study.yaml     # Machine-readable metadata (single source of truth)
            ├── CITATION.bib   # BibTeX entry
            ├── pyproject.toml # Study-local Python dependencies
            ├── paper/         # Compiled PDF, LaTeX/Typst source, figures
            └── review/        # Adversarial referee reports and revision logs
```

Each study's `study.yaml` acts as the structured source of truth for external catalogs and discovery frontends.

---

## Data Policy

Public-use summary outputs, regression tables, and lightweight crosswalks are committed directly. Large geospatial rasters/granules and restricted-redistribution microdata (such as IPUMS USA Census/ACS extracts, whose Terms of Use prohibit redistributing person-level microdata) are excluded, with exact API/download scripts documented inside each study's `README.md` instead.

---

## Licensing & Citation

- **Code & Scripts:** [MIT License](LICENSE-CODE)
- **Manuscripts, Figures & Derived Summary Tables:** [CC-BY-4.0](LICENSE-CONTENT)

See [`LICENSE`](LICENSE) for details and [`CITATION.cff`](CITATION.cff) (or each study's `CITATION.bib`) for citation entries.
