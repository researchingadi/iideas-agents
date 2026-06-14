from openai import OpenAI
import pandas as pd
import json
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

client = OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=token,
)

# ============================================
# STEP 1: LOAD CSV
# ============================================
def load_data(filepath):
    df = pd.read_csv(filepath)
    print("\n=== DATASET LOADED ===")
    print(f"Total buildings: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    return df

# ============================================
# STEP 2: AGENT PROPOSES RATIONALE
# ============================================
def propose_rationale(df, modifier=""):
    sample_data = df.head(10).to_string()
    columns = list(df.columns)

    prompt = f"""
    You are an expert building scientist working on archetype identification
    for disaster debris management and life cycle assessment.

    I have a dataset with {len(df)} buildings with these columns:
    {columns}

    Here is a sample of the data:
    {sample_data}

    {modifier}

    Based on this data, propose a clear rationale for classifying these
    buildings into archetypes. Include:
    1. Which columns to use for classification and why
    2. Specific criteria/thresholds for each factor
    3. How many archetypes you recommend and why
    4. The expected archetype categories

    Be specific, clear and explainable. Format your response clearly with
    numbered sections.
    """

    print("\n=== AGENT 1: ANALYZING YOUR DATASET ===")
    print("GPT-4o is proposing a classification rationale...\n")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert building scientist specializing in archetype identification for LCA and disaster debris management."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content

# ============================================
# STEP 3: USER REVIEWS RATIONALE
# ============================================
def user_review_rationale(rationale):
    print("=== PROPOSED CLASSIFICATION RATIONALE ===\n")
    print(rationale)
    print("\n" + "="*50)
    print("\nOptions:")
    print("1. Approve rationale and proceed")
    print("2. Request simpler rationale (fewer archetypes)")
    print("3. Request more detailed rationale (more archetypes)")
    print("4. Modify specific criteria")

    choice = input("\nEnter your choice (1/2/3/4): ").strip()
    return choice

# ============================================
# STEP 4: FEATURE PREPARATION (helper)
# ============================================
def _prepare_features(df, columns_to_use):
    """Convert selected columns to a float DataFrame suitable for StandardScaler."""
    data = df[columns_to_use].copy()

    for col in data.columns:
        numeric = pd.to_numeric(data[col], errors='coerce')
        if numeric.isna().sum() > data[col].isna().sum():
            cats = pd.Categorical(data[col].fillna('Unknown').astype(str))
            data[col] = cats.codes.astype(float)
        else:
            data[col] = numeric

    return data.fillna(0).astype(float)

# ============================================
# STEP 5: RUN ARCHETYPE IDENTIFICATION
# ============================================
def identify_archetypes(df, rationale, n_clusters=None):
    prompt = f"""
    Based on this rationale:
    {rationale}

    From these available columns: {list(df.columns)}

    Return ONLY a JSON object with:
    1. "columns_to_use": list of column names to use for clustering (only columns that exist in the dataset)
    2. "n_clusters": recommended number of archetypes (integer between 4 and 12)
    3. "archetype_names": list of descriptive names for each archetype (same length as n_clusters)

    Return ONLY valid JSON with no markdown, no code blocks, no explanation. Just the raw JSON object.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a data scientist. Return only raw valid JSON with no markdown formatting, no code blocks, no backticks."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response.choices[0].message.content.strip()
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    config = json.loads(raw)

    available_columns = list(df.columns)
    columns_to_use = [col for col in config["columns_to_use"] if col in available_columns]

    if not columns_to_use:
        columns_to_use = ['yearOfConstruction', 'grossFloorArea', 'numOfFloors', 'constructionType']
        columns_to_use = [col for col in columns_to_use if col in available_columns]

    if n_clusters is None:
        n_clusters = config["n_clusters"]

    archetype_names = config.get("archetype_names", [])
    if len(archetype_names) != n_clusters:
        archetype_names = [f"Archetype {i+1}" for i in range(n_clusters)]

    print(f"\n=== RUNNING ARCHETYPE IDENTIFICATION ===")
    print(f"Using columns: {columns_to_use}")
    print(f"Number of archetypes: {n_clusters}")
    print(f"Clustering all {len(df)} buildings...")

    # --- Cluster the full dataset ---
    all_data   = _prepare_features(df, columns_to_use)
    scaler     = StandardScaler()
    all_scaled = scaler.fit_transform(all_data)

    kmeans     = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    all_labels = kmeans.fit_predict(all_scaled)

    archetype_map = {i: archetype_names[i] for i in range(n_clusters)}
    df_labeled = df.copy()
    df_labeled['Archetype']      = all_labels
    df_labeled['Archetype_Name'] = df_labeled['Archetype'].map(archetype_map)

    # --- Stratified split: each archetype proportionally in both sets ---
    print(f"Stratified 80/20 split by archetype label...")
    df_train, df_test = train_test_split(
        df_labeled,
        test_size=0.2,
        random_state=42,
        stratify=df_labeled['Archetype']
    )

    # Slice the already-scaled matrix using the pre-reset original indices
    train_scaled = all_scaled[df_train.index]
    test_scaled  = all_scaled[df_test.index]
    sil_train    = silhouette_score(train_scaled, all_labels[df_train.index])
    sil_test     = silhouette_score(test_scaled,  all_labels[df_test.index])

    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    metrics = {
        "silhouette_score_train": round(float(sil_train), 4),
        "silhouette_score_test":  round(float(sil_test), 4),
        "inertia":                round(float(kmeans.inertia_), 2),
    }

    return df_train, df_test, archetype_names, n_clusters, metrics

# ============================================
# STEP 6: PRESENT RESULTS
# ============================================
def present_results(df_train, df_test, archetype_names, rationale, metrics):
    df_full  = pd.concat([df_train, df_test], ignore_index=True)
    n_train  = len(df_train)
    n_test   = len(df_test)
    n_total  = len(df_full)

    print("\n=== ARCHETYPE IDENTIFICATION RESULTS ===\n")
    print(f"{'Archetype':<40} {'Train':>8} {'Test':>8} {'Total':>8} {'%':>6}")
    print("-" * 75)

    missing_in_test = []
    for name in archetype_names:
        tr  = len(df_train[df_train['Archetype_Name'] == name])
        te  = len(df_test[df_test['Archetype_Name'] == name])
        tot = tr + te
        pct = tot / n_total * 100
        print(f"{name:<40} {tr:>8} {te:>8} {tot:>8} {pct:>5.1f}%")
        if te == 0:
            missing_in_test.append(name)

    print()
    if missing_in_test:
        print(f"  WARNING: These archetypes have NO test buildings: {missing_in_test}")
        print("           Archetype has too few buildings for stratified split — consider fewer archetypes.\n")

    results_summary = df_full.groupby('Archetype_Name').size().to_string()

    prompt = f"""
    We identified the following building archetypes from a community dataset of {n_total} buildings:
    {results_summary}

    Based on this rationale:
    {rationale}

    Provide a brief, clear explanation of:
    1. What each archetype represents
    2. Why this grouping makes sense for LCA and debris management
    3. Key characteristics of each group
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert building scientist. Be clear and concise."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("=== AGENT EXPLANATION ===")
    print(response.choices[0].message.content)

    # --- Validation summary ---
    sil_train = metrics["silhouette_score_train"]
    sil_test  = metrics["silhouette_score_test"]
    diff = abs(sil_train - sil_test)
    if diff < 0.05:
        generalization = "GOOD"
    elif diff < 0.15:
        generalization = "FAIR"
    else:
        generalization = "POOR"

    print("\n=== VALIDATION SUMMARY ===")
    print(f"Training buildings:        {n_train} (80%)")
    print(f"Test buildings:            {n_test} (20%)")
    print(f"Silhouette Score (Train):  {sil_train:.4f}  (higher is better, max 1.0)")
    print(f"Silhouette Score (Test):   {sil_test:.4f}")
    print(f"Inertia:                   {metrics['inertia']:.2f}")
    print(f"Model Generalization:      {generalization}")

    print("\n" + "="*50)
    print("\nOptions:")
    print("1. Approve archetypes and save results")
    print("2. Request fewer archetypes")
    print("3. Request more archetypes")
    print("4. Restart with modified rationale")

    choice = input("\nEnter your choice (1/2/3/4): ").strip()
    return choice

# ============================================
# STEP 7: SAVE RESULTS
# ============================================
def save_results(df_train, df_test, archetype_names, metrics):
    df_full = pd.concat([df_train, df_test], ignore_index=True)

    df_train.to_csv('archetype_results_train.csv', index=False)
    df_test.to_csv('archetype_results_test.csv',  index=False)
    df_full.to_csv('archetype_results_full.csv',  index=False)

    summary = {
        "total_buildings":        len(df_full),
        "train_buildings":        len(df_train),
        "test_buildings":         len(df_test),
        "total_archetypes":       len(archetype_names),
        "silhouette_score_train": metrics["silhouette_score_train"],
        "silhouette_score_test":  metrics["silhouette_score_test"],
        "inertia":                metrics["inertia"],
        "archetypes": []
    }

    for name in archetype_names:
        tr  = len(df_train[df_train['Archetype_Name'] == name])
        te  = len(df_test[df_test['Archetype_Name'] == name])
        tot = tr + te
        summary["archetypes"].append({
            "name":        name,
            "train_count": tr,
            "test_count":  te,
            "total_count": tot,
            "percentage":  round(tot / len(df_full) * 100, 1)
        })

    with open('archetype_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTS SAVED ===")
    print("archetype_results_train.csv  - Training buildings with archetype labels")
    print("archetype_results_test.csv   - Test buildings with archetype labels")
    print("archetype_results_full.csv   - All buildings combined")
    print("archetype_results.json       - Summary with train/test metrics for Agent 2")
    print(f"\nTotal archetypes identified: {len(archetype_names)}")
    print(f"Total buildings classified:  {len(df_full)}")

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("\n" + "="*50)
    print("  IIDEAS LAB - AGENT 1: BUILDING ARCHETYPE ID")
    print("="*50)

    filepath = input("\nEnter path to your CSV file: ").strip()
    df = load_data(filepath)

    rationale  = None
    n_clusters = None

    while True:
        if rationale is None:
            modifier = ""
            if n_clusters:
                modifier = f"Please aim for approximately {n_clusters} archetypes."
            rationale = propose_rationale(df, modifier)

        rationale_choice = user_review_rationale(rationale)

        if rationale_choice == "1":
            pass

        elif rationale_choice == "2":
            n_clusters = max(3, (n_clusters or 8) - 2)
            print(f"\nRequesting simpler rationale (~{n_clusters} archetypes)...")
            rationale = propose_rationale(df, f"Please use fewer archetypes, aim for {n_clusters}.")
            continue

        elif rationale_choice == "3":
            n_clusters = (n_clusters or 8) + 2
            print(f"\nRequesting more detailed rationale (~{n_clusters} archetypes)...")
            rationale = propose_rationale(df, f"Please use more archetypes, aim for {n_clusters}.")
            continue

        elif rationale_choice == "4":
            modification = input("\nWhat would you like to modify? ").strip()
            rationale = propose_rationale(df, f"User modification request: {modification}")
            continue

        # Cluster full dataset, then stratified split
        df_train_r, df_test_r, archetype_names, n_clusters, metrics = identify_archetypes(
            df.copy(), rationale, n_clusters
        )

        # Present results + validation summary
        result_choice = present_results(df_train_r, df_test_r, archetype_names, rationale, metrics)

        if result_choice == "1":
            save_results(df_train_r, df_test_r, archetype_names, metrics)
            print("\nAgent 1 complete! Ready for Agent 2.")
            break

        elif result_choice == "2":
            n_clusters = max(2, n_clusters - 2)
            print(f"\nRerunning with {n_clusters} archetypes...")

        elif result_choice == "3":
            n_clusters = n_clusters + 2
            print(f"\nRerunning with {n_clusters} archetypes...")

        elif result_choice == "4":
            rationale  = None
            n_clusters = None
            print("\nRestarting with modified rationale...")

if __name__ == "__main__":
    main()
