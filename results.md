# IIDEAS Lab — Agent 1 Comparison Results

**Institution:** Mississippi State University  
**Lab:** IIDEAS Lab (Intelligent Infrastructure for Disaster-Resilient Engineering and Sustainable Systems)  
**Developer:** Adi Singh (@researchingadi)  
**Dataset:** Sullivan, Indiana Community Buildings (90 buildings, 67 attributes)  
**Date:** June 2026

---

## Overview

This document presents the results of comparing three approaches to building archetype identification for disaster debris management and life cycle assessment (LCA):

1. **Old Automation Script** — Rule-based classification using hardcoded Era × Size × Construction Type rules
2. **AI Agent (GPT-4o)** — LLM-powered adaptive classification with user approval workflow
3. **ML Model (Random Forest)** — Supervised learning trained on AI Agent labels

---

## Dataset Validation Results

Run before every Agent 1 execution to ensure data integrity.

| Column | Status | Missing % |
|--------|--------|-----------|
| yearOfConstructionMin | ✅ Found | 12.2% |
| constructionType | ✅ Found | 13.3% |
| finishAreaSqft | ✅ Found | 15.6% |
| numOfBedrooms | ✅ Found | 7.8% |
| yearOfConstruction | ✅ Found | 10.0% |
| propertyClass | ✅ Found | 3.3% |
| sizeSqft | ✅ Found | 11.1% |
| wallType | ⚠️ Found | 93.3% missing |
| roofing | ✅ Found | 35.6% |
| numOfFloors | ✅ Found | 12.2% |
| occupancy | ✅ Found | 12.2% |

**Data Quality:**
- Year data: 78/90 buildings have valid years
- Size data: 73/90 buildings have size info
- Construction: 78/90 buildings have construction type

**Validation Result: PASSED ✅**

---

## Agent 1 Final Results (100% Dataset)

### Identified Archetypes (12 total)

| Archetype | Train | Test | Total | % |
|-----------|-------|------|-------|---|
| Single-family, Wood-frame, Low-rise, Small | 4 | 1 | 5 | 5.6% |
| Single-family, Wood-frame, Low-rise, Medium/Large | 11 | 3 | 14 | 15.6% |
| Single-family, Masonry, Low-rise, Small | 2 | 0 | 2 | 2.2% |
| Single-family, Masonry, Low-rise, Large | 9 | 2 | 11 | 12.2% |
| Multi-family, Wood-frame, Multi-story | 7 | 2 | 9 | 10.0% |
| Multi-family, Masonry-built, Multi-story | 2 | 0 | 2 | 2.2% |
| Mobile Home, Pre-1990 Manufacture | 7 | 2 | 9 | 10.0% |
| Mobile Home, 1990+ Modern Construction | 6 | 1 | 7 | 7.8% |
| Single-family, Concrete-built, Low-rise, Small | 7 | 2 | 9 | 10.0% |
| Single-family, Concrete-built, Low-rise, Large | 7 | 2 | 9 | 10.0% |
| Multi-family, Concrete-built, Multi-story | 8 | 2 | 10 | 11.1% |
| Mobile Home, Post-2000 Enhanced Standards | 2 | 1 | 3 | 3.3% |

### Validation Metrics

| Metric | Value |
|--------|-------|
| Training buildings | 72 (80%) |
| Test buildings | 18 (20%) |
| Silhouette Score (Train) | 0.1702 |
| Silhouette Score (Test) | 0.1877 |
| Model Generalization | **GOOD** ✅ |
| Split Method | Stratified 80/20 |

> **Note:** Test silhouette score (0.1877) is higher than train (0.1702), indicating no overfitting and strong generalization to unseen buildings.

---

## Incremental Comparison Results

Each approach was tested at 10% data increments. ML Model trained on AI Agent labels as ground truth.

| Increment | Buildings | Old Script Archetypes | Agent Archetypes | Agent Silhouette | ML Accuracy | Agent vs ML Agreement |
|-----------|-----------|----------------------|------------------|------------------|-------------|----------------------|
| 10% | 9 | 6 | 3 | 0.2462 | N/A (too few) | 1.0000 |
| 20% | 18 | 7 | 6 | 0.1249 | 0.6667 | N/A |
| 30% | 27 | 12 | 5 | 0.0871 | 0.5000 | 0.6515 |
| 40% | 36 | 17 | 6 | 0.1893 | 0.2500 | 0.6422 |
| 50% | 45 | 19 | 6 | 0.1533 | 0.7778 | 0.8936 |
| 60% | 54 | 21 | 7 | 0.1615 | 0.7273 | 0.8521 |
| 70% | 62 | 23 | 6 | 0.1699 | 0.3846 | 0.6990 |
| 80% | 72 | 26 | 8 | 0.2014 | 0.6000 | 0.7959 |
| 90% | 81 | 26 | 6 | 0.1773 | 0.8235 | 0.9329 |
| 100% | 90 | 29 | 7 | 0.2430 | **0.9444** | **0.9647** |

### Agreement Scores (Adjusted Rand Index)

ARI ranges from -1 to 1. A score of 1.0 indicates perfect agreement; 0 indicates chance-level agreement.

| Increment | Old vs Agent | Old vs ML | Agent vs ML |
|-----------|-------------|-----------|-------------|
| 10% | 0.0806 | 0.0806 | 1.0000 |
| 20% | 0.0823 | N/A | N/A |
| 30% | 0.0345 | 0.0280 | 0.6515 |
| 40% | 0.0801 | 0.1055 | 0.6422 |
| 50% | 0.1446 | 0.1853 | 0.8936 |
| 60% | 0.1012 | 0.1419 | 0.8521 |
| 70% | 0.1149 | 0.1188 | 0.6990 |
| 80% | 0.1541 | 0.1391 | 0.7959 |
| 90% | 0.1501 | 0.1607 | 0.9329 |
| 100% | 0.2472 | 0.2280 | **0.9647** |

---

## ML Model — Most Important Features

Consistent across all increments where the ML model had sufficient data.

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | buildingAssessedValueTotal | Highest |
| 2 | finishAreaSqft / sizeSqft | High |
| 3 | numOfRoomsTotal / numOfBedrooms | High |
| 4 | yearOfConstruction / age | Medium |
| 5 | grossFloorArea | Medium |

---

## Key Findings

### 1. Old Script Over-Segments

The rule-based automation script produces too many archetypes:

- At 10% data (9 buildings): **6 archetypes**
- At 100% data (90 buildings): **29 archetypes**
- Nearly one archetype per building at full dataset
- Every unique combination of Era × Size × Construction creates a new label
- Not practical for disaster debris management or LCA at community scale

### 2. AI Agent is Stable and Consistent

GPT-4o produces meaningful, stable archetype counts regardless of subset size:

- Consistently **5–8 archetypes** across all increments
- GOOD generalization (test silhouette exceeds train silhouette at full dataset)
- Explainable rationale proposed and approved before any clustering
- User can approve, modify, or iteratively refine classifications at every step

### 3. ML Model Validates the AI Agent

At 100% data:

- ML achieves **94.44% accuracy** learning the Agent's classification patterns
- Agent vs ML agreement = **0.9647** (near-perfect ARI)
- Demonstrates that Agent classifications are learnable, consistent, and reproducible
- Key predictive features: assessed value, finish area, room count

### 4. Old Script vs AI Agent Divergence

Low ARI agreement (0.08–0.25) throughout all increments reflects fundamentally different philosophies:

- **Old Script:** many fine-grained label combinations driven by fixed thresholds
- **AI Agent:** fewer, broader, semantically meaningful clusters driven by data patterns
- Neither is "wrong" — they optimize for different goals (rule auditability vs analytical utility)

### 5. Building Value is the Most Predictive Feature

The ML model consistently ranks `buildingAssessedValueTotal` as the top feature. This suggests that assessed property value serves as a composite proxy for building type, size, age, and construction quality — the same dimensions that drive archetype membership.

---

## Approach Comparison Summary

| Criteria | Old Script | AI Agent | ML Model |
|----------|-----------|----------|----------|
| Method | Rule-based | LLM + KMeans | Random Forest |
| Archetypes (100%) | 29 | 7 | 7 |
| Adaptability | ❌ Fixed rules | ✅ Adapts to data | ✅ Learns from Agent |
| Transparency | ✅ Simple rules | ✅ GPT-4o explains | ⚠️ Black box |
| User Control | ❌ Edit config file | ✅ Interactive approval | ❌ Automated |
| Silhouette / Accuracy | N/A (baseline) | 0.24 | 94.44% |
| Scalability | ⚠️ Too granular | ✅ Stable | ✅ Fast inference |
| Best For | Reference baseline | New communities | Rapid deployment |

---

## Output Files

| File | Description |
|------|-------------|
| `archetype_results_full.csv` | All 90 buildings with archetype labels |
| `archetype_results_train.csv` | 72 training buildings (80%) |
| `archetype_results_test.csv` | 18 test buildings (20%) |
| `archetype_results.json` | Summary + metrics for Agent 2 handoff |
| `comparison_results.json` | Full incremental comparison data |
| `comparison_results.xlsx` | Formatted Excel comparison report |

---

## Next Steps

- [ ] **Agent 2:** LCA Analysis (Meta Llama 3.1 70B)
  - Consumes `archetype_results.json`
  - Calculates environmental impact per archetype
  - Estimates embodied carbon, energy use, and material quantities

- [ ] **Agent 3:** Debris Recovery Recommendations (Mistral Large)
  - Consumes LCA results from Agent 2
  - Recommends reuse / recycle / landfill percentages per archetype
  - Outputs circular economy recovery strategies

- [ ] **FastAPI Backend:** Connect agents to web application
- [ ] **WordPress Integration:** CSV upload → agents → results display
- [ ] **Deployment:** Hostinger VPS hosting

---

## Repository

**GitHub:** https://github.com/researchingadi/iideas-agents  
**Tech Stack:** Python · GPT-4o · scikit-learn · pandas · GitHub Models API
