# IIDEAS Lab AI Agents
### Intelligent Pipeline for Disaster Debris Management & Life Cycle Assessment

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Status-Active%20Research-orange" />
  <img src="https://img.shields.io/badge/Institution-Mississippi%20State%20University-990000" />
  <img src="https://img.shields.io/badge/Models-GPT--4o%20%7C%20Llama%203.1%20%7C%20Mistral-blueviolet" />
</p>

---

## Overview

This repository presents a multi-agent AI pipeline developed at the **IIDEAS Lab (Intelligent Infrastructure for Disaster-Resilient Engineering and Sustainable Systems)** at **Mississippi State University** by **Adi Singh**. The system orchestrates a sequential chain of three large language model (LLM) agents to automate the analysis of community building inventories for **disaster debris management** and **life cycle assessment (LCA)**. By combining the reasoning capabilities of frontier models — GPT-4o, Meta Llama 3.1 70B, and Mistral Large — with classical machine learning methods, the pipeline transforms raw building data into actionable recovery intelligence. This work contributes to the growing field of AI-assisted disaster resilience engineering, where explainability, human-in-the-loop validation, and scientific rigor are first-class requirements.

---

## Agent Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IIDEAS Lab AI Agent Pipeline                     │
└─────────────────────────────────────────────────────────────────────┘

  Community Building Dataset (CSV)
  90 buildings · 67 attributes
           │
           ▼
┌──────────────────────┐
│      AGENT 1         │  ← GPT-4o (GitHub Models)
│  Building Archetype  │
│   Identification     │
│                      │
│  • Rationale design  │
│  • KMeans clustering │
│  • Stratified split  │
│  • Silhouette scores │
└──────────┬───────────┘
           │  archetype_results.json
           ▼
┌──────────────────────┐
│      AGENT 2         │  ← Meta Llama 3.1 70B (GitHub Models)
│    LCA Analysis      │  ⏳ Coming Soon
│                      │
│  • Material takeoffs │
│  • Impact per type   │
│  • Embodied carbon   │
│  • Impact reports    │
└──────────┬───────────┘
           │  lca_results.json
           ▼
┌──────────────────────┐
│      AGENT 3         │  ← Mistral Large (GitHub Models)
│  Debris Management   │  ⏳ Coming Soon
│  & Recovery          │
│                      │
│  • Recovery strategy │
│  • Reuse / Recycle   │
│  • Disposal routing  │
│  • Circular economy  │
└──────────────────────┘
```

---

## Features

- **LLM-Driven Rationale** — GPT-4o reads the dataset and proposes an explainable, scientifically grounded classification rationale before any clustering occurs
- **Human-in-the-Loop Approval** — every rationale and result set is reviewed and can be iteratively refined by the user before proceeding
- **Robust Clustering** — KMeans with `n_init=10` and StandardScaler normalization; string columns encoded via categorical codes; mixed-type columns handled with `pd.to_numeric` coercion
- **Stratified Validation** — 80/20 train/test split stratified by archetype label guarantees proportional representation of every cluster in both sets
- **Generalization Metrics** — silhouette scores computed independently on train and test subsets; gap rated GOOD / FAIR / POOR
- **Structured Handoff** — each agent writes a JSON summary consumed by the next, enabling modular, resumable execution
- **Secure Credential Handling** — API tokens loaded from `.env`; no credentials in source

---

## Tech Stack

| Layer | Technology |
|---|---|
| 🐍 Language | Python 3.14 |
| 🤖 LLM — Agent 1 | GPT-4o via GitHub Models Marketplace |
| 🦙 LLM — Agent 2 | Meta Llama 3.1 70B via GitHub Models Marketplace |
| 🌊 LLM — Agent 3 | Mistral Large via GitHub Models Marketplace |
| 📊 Clustering | scikit-learn (KMeans, StandardScaler, silhouette_score) |
| 🔀 Validation | sklearn.model_selection.train_test_split |
| 🗃️ Data | pandas, numpy |
| 🔐 Secrets | python-dotenv |
| 📤 Output | CSV + JSON |

---

## Dataset

The pilot dataset covers the **Sullivan, Indiana** community and was compiled for disaster debris estimation and recovery planning research.

| Property | Value |
|---|---|
| Community | Sullivan, Indiana |
| Buildings | 90 |
| Attributes | 67 |
| Key columns | Construction type, year built, gross floor area, number of floors, wall material, roofing type, foundation type, auxiliary structures |
| Format | CSV |

---

## Installation

### Prerequisites

- Python 3.10 or later
- A [GitHub Models](https://github.com/marketplace/models) personal access token (free with a GitHub account)

### 1. Clone the repository

```bash
git clone https://github.com/researchingadi/iideas-agents.git
cd iideas-agents
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install openai pandas numpy scikit-learn python-dotenv openpyxl
```

### 4. Configure your API token

Create a `.env` file in the project root:

```bash
GITHUB_TOKEN=your_github_personal_access_token_here
```

> Your token needs the `models:read` scope. Generate one at **GitHub → Settings → Developer settings → Personal access tokens**.

---

## Usage

### Running Agent 1

```bash
python agent1_archetype.py
```

When prompted, enter the path to your building dataset:

```
Enter path to your CSV file: community_data.csv
```

#### Interactive workflow

```
==================================================
  IIDEAS LAB - AGENT 1: BUILDING ARCHETYPE ID
==================================================

=== DATASET LOADED ===
Total buildings: 90
...

=== AGENT 1: ANALYZING YOUR DATASET ===
GPT-4o is proposing a classification rationale...

=== PROPOSED CLASSIFICATION RATIONALE ===

1. Columns selected for classification: constructionType, yearOfConstruction,
   grossFloorArea, numOfFloors, wallMaterial, roofingType ...
2. Criteria and thresholds: ...
3. Recommended archetypes: 8 ...

Options:
1. Approve rationale and proceed
2. Request simpler rationale (fewer archetypes)
3. Request more detailed rationale (more archetypes)
4. Modify specific criteria

Enter your choice (1/2/3/4):
```

After approval, the agent clusters the full dataset, stratifies the split, and presents results:

```
=== ARCHETYPE IDENTIFICATION RESULTS ===

Archetype                                Train     Test    Total      %
---------------------------------------------------------------------------
Pre-War Masonry Residential                  9        2       11   12.2%
Mid-Century Wood Frame                      10        3       13   14.4%
Post-War Brick Ranch                         8        2       10   11.1%
...

=== VALIDATION SUMMARY ===
Training buildings:        72 (80%)
Test buildings:            18 (20%)
Silhouette Score (Train):  0.3009  (higher is better, max 1.0)
Silhouette Score (Test):   0.2464
Inertia:                   412.37
Model Generalization:      FAIR
```

### Output files

| File | Description |
|---|---|
| `archetype_results_train.csv` | 72 training buildings with archetype labels |
| `archetype_results_test.csv` | 18 test buildings with archetype labels |
| `archetype_results_full.csv` | All 90 buildings combined |
| `archetype_results.json` | Structured summary consumed by Agent 2 |

---

## Results — Agent 1 (Sullivan, IN)

| Metric | Value |
|---|---|
| Total buildings | 90 |
| Archetypes identified | 8 |
| Clustering algorithm | KMeans (`n_init=10`, `random_state=42`) |
| Train / Test split | 72 / 18 (stratified) |
| Silhouette Score — Train | **0.3009** |
| Silhouette Score — Test | **0.2464** |
| Inertia | 412.37 |
| Model Generalization | **FAIR** |

> Silhouette scores in the 0.25–0.35 range are typical for real-world heterogeneous building inventories where clusters overlap in multiple feature dimensions. The stratified split confirms consistent cluster geometry between training and test subsets.

---

## Project Structure

```
iideas-agents/
│
├── agent1_archetype.py        # Agent 1: Building Archetype Identification
├── agent2_lca.py              # Agent 2: LCA Analysis (coming soon)
├── agent3_debris.py           # Agent 3: Debris Management (coming soon)
│
├── community_data.csv         # Sullivan, IN building inventory
│
├── archetype_results_train.csv   # Agent 1 output — training set
├── archetype_results_test.csv    # Agent 1 output — test set
├── archetype_results_full.csv    # Agent 1 output — full dataset
├── archetype_results.json        # Agent 1 → Agent 2 handoff
│
├── .env                       # API token (not committed)
├── .gitignore
└── README.md
```

---

## Roadmap

- [x] Agent 1 — Building Archetype Identification (GPT-4o)
- [x] Stratified train/test validation with silhouette scoring
- [x] Human-in-the-loop iterative refinement workflow
- [ ] Agent 2 — LCA Analysis (Meta Llama 3.1 70B)
- [ ] Agent 3 — Debris Management & Recovery (Mistral Large)
- [ ] End-to-end pipeline runner (`run_pipeline.py`)
- [ ] Support for multi-community batch processing
- [ ] Web dashboard for non-technical stakeholders
- [ ] Integration with FEMA debris estimation guidelines
- [ ] Peer-reviewed publication of methodology and results

---

## Citation

If you use this codebase or methodology in your research, please cite:

```bibtex
@software{singh2025iideas,
  author       = {Singh, Adi},
  title        = {IIDEAS Lab AI Agents: An LLM Pipeline for Disaster
                  Debris Management and Life Cycle Assessment},
  year         = {2025},
  institution  = {Mississippi State University, IIDEAS Lab},
  url          = {https://github.com/researchingadi/iideas-agents},
  note         = {Supervised by Dr. Mojtaba Parsaee}
}
```

---

## Authors

| Name | Role | GitHub |
|---|---|---|
| Adi Singh | Developer | [@researchingadi](https://github.com/researchingadi) |
| Dr. Mojtaba Parsaee | Principal Investigator | Mississippi State University |

---

## Acknowledgements

This research is conducted at the **IIDEAS Lab** (Intelligent Infrastructure for Disaster-Resilient Engineering and Sustainable Systems) at **Mississippi State University**.

LLM access is provided through the **[GitHub Models Marketplace](https://github.com/marketplace/models)** educational program, enabling free access to GPT-4o, Meta Llama 3.1, and Mistral Large for research purposes.

---

## License

This project is licensed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<p align="center">
  Built with ♥ at IIDEAS Lab · Mississippi State University
</p>
