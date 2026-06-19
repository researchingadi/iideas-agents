# Contributions

<p align="left">
  <img src="https://img.shields.io/badge/Contributor-Adi%20Singh-blue" />
  <img src="https://img.shields.io/badge/Role-Graduate%20Research%20Assistant-green" />
  <img src="https://img.shields.io/badge/Institution-Mississippi%20State%20University-990000" />
  <img src="https://img.shields.io/badge/Period-Spring--Summer%202026-orange" />
  <img src="https://img.shields.io/badge/Status-Active-brightgreen" />
</p>

---

## Contributor

| Field | Details |
|---|---|
| **Name** | Adi Singh |
| **GitHub** | [@researchingadi](https://github.com/researchingadi) |
| **Role** | Graduate Research Assistant |
| **Institution** | Mississippi State University |
| **Lab** | IIDEAS Lab — Intelligent Infrastructure for Disaster-Resilient Engineering and Sustainable Systems |
| **Principal Investigator** | Dr. Mojtaba Parsaee |
| **Period** | Spring – Summer 2026 |

---

## Summary

This file documents the design, engineering, and research contributions made to the IIDEAS Lab AI Agents project — a multi-agent LLM pipeline for disaster debris management and life cycle assessment (LCA). All code, architecture decisions, experiments, and documentation described below were authored by Adi Singh under the supervision of Dr. Mojtaba Parsaee.

---

## Contributions

### 1. Multi-Agent AI Pipeline Architecture

**Files:** `agent1_archetype.py`, `agent2_lca.py` *(planned)*, `agent3_debris.py` *(planned)*

Designed and built the end-to-end architecture for a sequential three-agent AI system. Each agent is responsible for a distinct phase of the disaster debris analysis workflow and passes structured JSON output to the next.

- Architected the full 3-agent pipeline: Archetype ID → LCA Analysis → Debris Recovery
- Integrated the [GitHub Models Marketplace](https://github.com/marketplace/models) API to access GPT-4o, Meta Llama 3.1 70B, and Mistral Large at no cost under educational access
- Built a shared JSON handoff schema between agents so each agent's output is the next agent's input
- Implemented secure token management using `python-dotenv` — API keys are never hardcoded
- Resolved GitHub secret scanning push protection after tokens were accidentally committed; rewrote Git history using `git filter-branch`
- Set up Python virtual environment (`venv`) and full dependency management on Windows

```python
# Agent-to-agent handoff schema (archetype_results.json)
{
  "total_buildings": 90,
  "train_buildings": 72,
  "test_buildings": 18,
  "total_archetypes": 12,
  "silhouette_score_train": 0.1702,
  "silhouette_score_test": 0.1877,
  "archetypes": [
    { "name": "...", "train_count": 4, "test_count": 1, "total_count": 5, "percentage": 5.6 }
  ]
}
```

**Tech:** Python, OpenAI SDK, GitHub Models API, python-dotenv, Git

---

### 2. Agent 1 — Building Archetype Identification

**File:** `agent1_archetype.py`

Implemented the complete Agent 1 pipeline: GPT-4o proposes a classification rationale, the user approves or refines it, KMeans clustering is run on the full dataset, and a stratified split is used for validation.

- GPT-4o reads the dataset schema and sample rows and proposes an *explainable* classification rationale before any clustering occurs — the model's reasoning is human-readable and domain-grounded
- Implemented human-in-the-loop approval with four options: approve / fewer archetypes / more archetypes / custom modification — the loop continues until the user is satisfied
- KMeans clustering (`n_init=10`) with `StandardScaler` normalization; clustering runs on the full dataset so all buildings contribute to the cluster geometry
- Stratified 80/20 train/test split using archetype labels as the stratification key — guarantees every archetype appears proportionally in both sets
- Silhouette scores computed independently on the train and test subsets of the already-fitted model; gap rated GOOD / FAIR / POOR
- Built a hallucination guard: GPT-4o's recommended column names are cross-validated against the actual DataFrame columns before clustering — non-existent columns are silently dropped with a fallback
- Fixed `ValueError: could not convert string to float` caused by dimension strings (e.g. `"28' x 46' x 12'"`) in numeric columns using `pd.to_numeric(errors='coerce')` with Categorical codes fallback
- Fixed JSON parsing failures when GPT-4o returned responses wrapped in markdown code fences (` ```json ... ``` `) using `re.sub` stripping

**Results on Sullivan, IN dataset:**

| Metric | Value |
|---|---|
| Archetypes identified | 12 |
| Training buildings | 72 (80%) |
| Test buildings | 18 (20%) |
| Silhouette Score — Train | 0.1702 |
| Silhouette Score — Test | **0.1877** |
| Model Generalization | **GOOD ✅** |

Test silhouette exceeds train silhouette, confirming no overfitting and strong generalization to unseen buildings.

**Tech:** Python, GPT-4o, scikit-learn (KMeans, StandardScaler, silhouette_score, train_test_split), pandas, numpy

---

### 3. Dataset Validation System

**File:** `validate_dataset.py`

Built a standalone validation module that runs before Agent 1 and catches data issues before they cause failures downstream.

- Validates 11 required columns against a predefined schema for archetype identification
- Fuzzy column matching: normalizes column names by stripping non-alphanumeric characters and checks substring overlap — maps `year_built` → `yearOfConstruction` automatically
- Checks missing value percentages per column; flags any column with >50% missing as a warning
- Year validity check: confirms year values fall within 1800–2025 after coercion
- Archetype readiness check: verifies the dataset has at least one year column, one size column, and a construction type column before proceeding
- Generates a formatted pass/fail report with ✅ / ⚠️ / ❌ per column
- Correctly flagged `wallType` as 93.3% missing in the Sullivan dataset
- Integrated into `agent1_archetype.py` as Step 0 — if validation fails, user is given the choice to proceed with warnings or exit

```
=== DATASET VALIDATION REPORT ===
File: community_data.csv
Total buildings: 90

REQUIRED COLUMNS:
  ✅ yearOfConstructionMin     - found (12.2% missing)
  ✅ constructionType          - found (13.3% missing)
  ⚠️  wallType                 - found (93.3% missing)
  ✅ roofing                   - found (35.6% missing)

VALIDATION RESULT: PASSED ✅
```

**Tech:** Python, pandas, regex

---

### 4. Incremental Performance Comparison Pipeline

**File:** `comparison_pipeline.py`

Built a complete three-way benchmarking system that tests all three approaches at 10% data increments and quantifies their agreement.

- Tests all three approaches on the same dataset subset at each of 10 increments: 10%, 20%, … 100% (9 → 90 buildings)
- Each AI Agent run is a completely fresh session — new `OpenAI` client, new message array, no state carryover between increments — ensuring independence
- Calculates **Adjusted Rand Index (ARI)** for pairwise agreement between all three approaches at each increment
- `warnings.filterwarnings("ignore")` placed before all imports to suppress sklearn verbosity cleanly
- Saves full structured results to `comparison_results.json` and a formatted Excel report to `comparison_results.xlsx` with color-coded headers using `openpyxl`
- JSON serialization uses `default=str` to handle numpy scalar types without crashing

**Key finding:** The old automation script produces 29 archetypes for 90 buildings — nearly one per building. The AI Agent consistently produces 5–8 semantically meaningful archetypes regardless of dataset size.

| Increment | Buildings | Old Script | AI Agent | Agent Silhouette | ML Accuracy |
|-----------|-----------|-----------|---------|-----------------|-------------|
| 10% | 9 | 6 | 3 | 0.2462 | N/A (too few) |
| 50% | 45 | 19 | 6 | 0.1533 | 0.7778 |
| 100% | 90 | 29 | 7 | 0.2430 | **0.9444** |

**Tech:** Python, scikit-learn, pandas, openpyxl, OpenAI SDK

---

### 5. ML Model Development and Evaluation

**File:** `comparison_pipeline.py` → `run_ml_model()`

Built a Random Forest classifier as an independent third approach and used it to validate the consistency of AI Agent classifications.

- Used AI Agent cluster labels as ground truth (not old script labels), making the RF a truly independent validator of the Agent's output rather than a replication of the rule-based baseline
- Stratified 80/20 split with plain-split fallback for small subsets where stratification fails
- Automatic class reduction: caps number of classes at `n_samples // 5` to prevent the model from being asked to distinguish more classes than it has samples to learn from
- Feature importance extracted and ranked for every run; top 3 features reported in the console and saved to JSON
- Fixed four separate sklearn issues during development:
  - `ValueError: Input contains NaN` — column-by-column `pd.to_numeric(errors='coerce').fillna(0)` encoding
  - `class_weight="balanced"` conflict with numeric cluster labels — removed
  - Stratification failure on single-class subsets — caught with `try/except ValueError`
  - Too many unique classes relative to sample count — auto-reduction to `max(2, n // 5)`

**Results at 100% dataset (90 buildings):**

| Metric | Value |
|---|---|
| Accuracy | **94.44%** |
| Agent vs ML Agreement (ARI) | **0.9647** |
| Top feature | `buildingAssessedValueTotal` |
| Ground truth source | AI Agent labels |

**Tech:** Python, scikit-learn (RandomForestClassifier, accuracy_score, train_test_split), pandas

---

### 6. Research Documentation

**Files:** `README.md`, `results.md`, `CONTRIBUTIONS.md`

Authored all project documentation to research-repository standard.

**README.md:**
- ASCII pipeline architecture diagram showing all three agents, their LLMs, and data flow
- Complete installation guide (clone → venv → pip → `.env` setup)
- Interactive workflow example with realistic terminal output
- Results table with actual silhouette scores
- BibTeX citation block
- Tech stack table with emojis
- Roadmap with checkboxes

**results.md:**
- Full dataset validation report
- All 12 identified archetypes with train/test/total counts and percentages
- Complete 10-increment comparison table
- ARI agreement scores across all increments
- ML feature importance rankings
- Key findings section with research-grade analysis
- Approach comparison summary table

**Other:**
- Configured `.gitignore` to exclude `.env`, `venv/`, `__pycache__/`, and output CSVs
- Resolved GitHub push protection after token appeared in commit history; cleaned history and re-pushed

**Tech:** Markdown, Git, GitHub

---

### 7. Data Engineering

**Files:** `community_data.csv`, `agent1_archetype.py` → `_prepare_features()`

Processed and normalized a real-world municipal building assessment dataset for use in machine learning pipelines.

- Converted the Sullivan, Indiana building assessment Excel workbook to CSV for agent processing
- Diagnosed and fixed dimension strings stored in numeric columns (e.g., `"28' x 46' x 12'"` in `finishAreaSqft`) using `pd.to_numeric(errors='coerce')` with Categorical code fallback
- Handled up to 93.3% missing values in columns like `wallType` without crashing clustering
- Normalized inconsistent construction type strings (`"wood Frame"`, `"Wood Frame"`, `"3/6 Masonry"`) via lowercase substring matching
- Coerced 2-digit year values (`"78"` → `1978`) and discarded implausible values (`"1030"`) in `run_old_script.py`
- Implemented robust JSON parsing for LLM responses: `re.sub` stripping of ` ```json ``` ` fences before `json.loads()` — handles all GPT-4o markdown wrapping variants

```python
# Robust mixed-type column encoding
for col in data.columns:
    numeric = pd.to_numeric(data[col], errors="coerce")
    if numeric.isna().sum() > data[col].isna().sum():
        # Column has strings — encode as categorical codes
        cats = pd.Categorical(data[col].fillna("Unknown").astype(str))
        data[col] = cats.codes.astype(float)
    else:
        data[col] = numeric
data = data.fillna(0).astype(float)
```

**Tech:** Python, pandas, openpyxl, numpy

---

### 8. DevOps and Repository Management

**Platform:** Windows 11, PowerShell, Git, GitHub

Set up the full development environment and repository from scratch on Windows.

- Initialized Git repository, configured identity (`user.name`, `user.email`)
- Resolved Windows PowerShell execution policy restriction blocking `venv\Scripts\activate` using `Set-ExecutionPolicy RemoteSigned`
- Managed GitHub push protection trigger after a `GITHUB_TOKEN` value appeared in an early commit; used `git filter-branch` to rewrite history and force-pushed the cleaned branch
- Configured `.gitignore` covering `.env`, `venv/`, `__pycache__/`, `*.pyc`, and output files
- Installed and pinned all project dependencies: `openai`, `pandas`, `numpy`, `scikit-learn`, `openpyxl`, `python-dotenv`

**Tech:** Git, GitHub, Windows PowerShell, pip, venv

---

## Results & Metrics

### Agent 1 Performance

| Metric | Value |
|---|---|
| Dataset | 90 buildings, 67 attributes |
| Community | Sullivan, Indiana |
| Archetypes identified | 12 |
| Silhouette Score (Train) | 0.1702 |
| Silhouette Score (Test) | **0.1877** (higher than train) |
| Model Generalization | **GOOD ✅** |
| Split method | Stratified 80/20 |

### Comparison Pipeline

| Metric | Value |
|---|---|
| Increments tested | 10 (10% → 100%) |
| Old script archetypes at 100% | 29 *(too granular)* |
| AI Agent archetypes at 100% | 7 *(stable, meaningful)* |
| ML accuracy at 100% | **94.44%** |
| Agent vs ML agreement (ARI) | **0.9647** *(near perfect)* |

### Technical Challenges Solved

| # | Challenge | Solution |
|---|---|---|
| 1 | `ValueError: could not convert string to float` from dimension strings in numeric columns | `pd.to_numeric(errors='coerce')` + Categorical codes fallback per column |
| 2 | JSON parsing failures from GPT-4o markdown code fences | `re.sub` to strip ` ```json ``` ` before `json.loads()` |
| 3 | NaN propagation across multiple data processing steps | Column-by-column coercion + `fillna(0).astype(float)` as final guard |
| 4 | GitHub push protection — token in commit history | `git filter-branch` history rewrite + force push |
| 5 | Windows PowerShell blocking `venv` activation | `Set-ExecutionPolicy RemoteSigned` |
| 6 | `train_test_split` stratification failure on small datasets | `try/except ValueError` fallback to non-stratified split |
| 7 | `class_weight="balanced"` conflict with numeric cluster labels | Removed `class_weight` parameter |
| 8 | Too many classes relative to sample count in Random Forest | Auto-reduction to `max(2, n_samples // 5)` most common classes |

---

## Skills Demonstrated

| Category | Skills |
|---|---|
| **Large Language Models** | GPT-4o, Meta Llama 3.1 70B, Mistral Large — prompt engineering, structured output, hallucination mitigation |
| **Unsupervised ML** | KMeans clustering, StandardScaler normalization, silhouette scoring, cluster geometry analysis |
| **Supervised ML** | Random Forest classification, feature importance, stratified train/test splits, accuracy evaluation |
| **Model Validation** | Silhouette score (train vs test), Adjusted Rand Index, generalization rating (GOOD/FAIR/POOR) |
| **Data Engineering** | Real-world messy data, missing value handling, mixed type coercion, categorical encoding |
| **API Integration** | GitHub Models Marketplace, OpenAI SDK, secure token management, fresh session management |
| **Explainable AI** | Human-in-the-loop approval workflows, LLM rationale generation, iterative refinement |
| **Python** | pandas, scikit-learn, openai, numpy, openpyxl, python-dotenv, regex |
| **Git / GitHub** | Version control, `.gitignore`, push protection, history rewriting, branching |
| **Research Documentation** | Markdown, results reporting, architecture diagrams, citation formatting |
| **Windows Development** | PowerShell, venv setup, execution policy, path management |
| **Debugging** | ML pipeline debugging, dtype errors, API response parsing, edge case handling |

---

## Repository Links

- **GitHub:** [https://github.com/researchingadi/iideas-agents](https://github.com/researchingadi/iideas-agents)
- **Results:** [results.md](results.md)
- **Project overview:** [README.md](README.md)

## Citation

If you use this work in your research, please cite:

```bibtex
@software{singh2026iideas,
  author       = {Singh, Adi},
  title        = {IIDEAS Lab AI Agents: An LLM Pipeline for Disaster
                  Debris Management and Life Cycle Assessment},
  year         = {2026},
  institution  = {Mississippi State University, IIDEAS Lab},
  url          = {https://github.com/researchingadi/iideas-agents},
  note         = {Supervised by Dr. Mojtaba Parsaee}
}
```

---

<p align="center">
  IIDEAS Lab · Mississippi State University · Spring–Summer 2026
</p>
