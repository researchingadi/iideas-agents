# IIDEAS Lab — AI Agent & ML Model Setup Instructions

## Files Included

- `agent1_archetype.py` — AI Agent for building archetype identification (GPT-4o)
- `validate_dataset.py` — Validates dataset before processing
- `ml_model.py` — Random Forest model that learns from Agent 1's classifications
- `community_data.csv` — Sullivan, Indiana building dataset
- `requirements.txt` — Python dependencies
- `.env` — Configuration file for your GitHub token

---

## Step 1: Install Python Dependencies

Open a terminal in this folder and run:

```
pip install -r requirements.txt
```

---

## Step 2: Get a GitHub Personal Access Token

This is required to access GPT-4o for free through GitHub Models Marketplace.

1. Go to github.com → click your profile picture → Settings
2. Scroll down to Developer Settings (bottom left)
3. Click Personal Access Tokens → Tokens (classic)
4. Click Generate new token (classic)
5. Name it anything (e.g. "iideas-agent")
6. Set expiration to 90 days
7. Leave all permission checkboxes unchecked
8. Click Generate token
9. Copy the token immediately (you only see it once)

---

## Step 3: Add Your Token

Open the `.env` file in this folder and replace the placeholder:

```
GITHUB_TOKEN=your_github_token_here
```

with your actual token:

```
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxx
```

Save the file.

---

## Step 4: Run the Scripts (In This Order)

### A) Validate the Dataset

```
python validate_dataset.py
```

This checks that `community_data.csv` has all the required columns and reports any data quality issues before proceeding.

### B) Run the AI Agent

```
python agent1_archetype.py
```

When prompted, enter: `community_data.csv`

This will:
1. Run validation automatically
2. GPT-4o will analyze the dataset and propose a classification rationale
3. You will be asked to approve, request fewer/more archetypes, or modify criteria
4. Once approved, it runs clustering and shows you the identified archetypes with a stratified 80/20 split
5. You approve the final results
6. This creates: `archetype_results_full.csv`, `archetype_results_train.csv`, `archetype_results_test.csv`, `archetype_results.json`

### C) Run the ML Model

```
python ml_model.py
```

This automatically loads `archetype_results_full.csv` (created in step B) and:
1. Trains a Random Forest classifier on 80% of the data
2. Tests it on the remaining 20%
3. Reports accuracy and the most important features
4. Saves predictions to `ml_predictions.csv`

---

## What Each Script Does

### validate_dataset.py
Checks the uploaded CSV file has all required columns (year built, size, construction type, etc.) before any processing happens. Flags missing or low-quality data.

### agent1_archetype.py
Uses GPT-4o to intelligently classify buildings into archetypes. Unlike hardcoded rules, the AI proposes its own classification rationale based on the actual data, which you can review and approve before it runs. Includes built-in hallucination checks to ensure the AI is not making up data, and validates that every building gets properly classified with no data loss.

### ml_model.py
Takes the archetypes identified by the AI Agent and trains a separate, traditional Random Forest model to see if it can learn to predict the same archetypes using only basic building features. High accuracy here validates that the Agent's classifications are consistent and not random.

---

## Troubleshooting

**"Bad credentials" error:**  
Your GitHub token in `.env` is missing, incorrect, or expired. Generate a new one following Step 2.

**"Module not found" error:**  
Run `pip install -r requirements.txt` again.

**Validation fails:**  
Check the validation report output — it will tell you exactly which columns are missing or have too much missing data.

---

## Output Files Reference

| File | Created By | Description |
|------|------------|-------------|
| `archetype_results_full.csv` | agent1_archetype.py | All buildings with archetype labels |
| `archetype_results_train.csv` | agent1_archetype.py | 80% training subset |
| `archetype_results_test.csv` | agent1_archetype.py | 20% test subset |
| `archetype_results.json` | agent1_archetype.py | Summary with validation metrics |
| `ml_predictions.csv` | ml_model.py | ML model predictions per building |

---

## Questions

Contact Adi Singh ([@researchingadi](https://github.com/researchingadi)) for any setup issues.
