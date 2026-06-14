from openai import OpenAI
import pandas as pd
import json
import re
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
# STEP 4: RUN ARCHETYPE IDENTIFICATION
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

    # Strip markdown code fences (e.g. ```json ... ``` or ``` ... ```)
    raw = re.sub(r'^```(?:json)?\s*', '', raw, flags=re.MULTILINE)
    raw = re.sub(r'```\s*$', '', raw, flags=re.MULTILINE)
    raw = raw.strip()

    config = json.loads(raw)

    # Only use columns that actually exist in the dataframe
    available_columns = list(df.columns)
    columns_to_use = [col for col in config["columns_to_use"] if col in available_columns]

    if not columns_to_use:
        # Fallback columns if none match
        columns_to_use = ['yearOfConstruction', 'grossFloorArea', 'numOfFloors', 'constructionType']
        columns_to_use = [col for col in columns_to_use if col in available_columns]

    if n_clusters is None:
        n_clusters = config["n_clusters"]

    # Make sure archetype names match cluster count
    archetype_names = config.get("archetype_names", [])
    if len(archetype_names) != n_clusters:
        archetype_names = [f"Archetype {i+1}" for i in range(n_clusters)]

    print(f"\n=== RUNNING ARCHETYPE IDENTIFICATION ===")
    print(f"Using columns: {columns_to_use}")
    print(f"Number of archetypes: {n_clusters}")

    # Prepare data
    cluster_data = df[columns_to_use].copy()

    # Convert every column to numeric; fall back to categorical codes
    # when pd.to_numeric introduces new NaN values (i.e. the column has strings)
    for col in cluster_data.columns:
        numeric = pd.to_numeric(cluster_data[col], errors='coerce')
        if numeric.isna().sum() > cluster_data[col].isna().sum():
            cluster_data[col] = pd.Categorical(
                cluster_data[col].fillna('Unknown').astype(str)
            ).codes.astype(float)
        else:
            cluster_data[col] = numeric

    # Fill any remaining NaN with 0 and guarantee float dtype
    cluster_data = cluster_data.fillna(0).astype(float)

    # Normalize
    scaler = StandardScaler()
    cluster_data_scaled = scaler.fit_transform(cluster_data)

    # Run KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Archetype'] = kmeans.fit_predict(cluster_data_scaled)

    # Map to names
    archetype_map = {i: archetype_names[i] for i in range(n_clusters)}
    df['Archetype_Name'] = df['Archetype'].map(archetype_map)

    return df, archetype_names, n_clusters

# ============================================
# STEP 5: PRESENT RESULTS
# ============================================
def present_results(df, archetype_names, rationale):
    print("\n=== ARCHETYPE IDENTIFICATION RESULTS ===\n")

    for name in archetype_names:
        archetype_df = df[df['Archetype_Name'] == name]
        print(f"Archetype: {name}")
        print(f"  Buildings: {len(archetype_df)} ({len(archetype_df)/len(df)*100:.1f}%)")
        print()

    results_summary = df.groupby('Archetype_Name').size().to_string()

    prompt = f"""
    We identified the following building archetypes from a community dataset of {len(df)} buildings:
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

    print("\n" + "="*50)
    print("\nOptions:")
    print("1. Approve archetypes and save results")
    print("2. Request fewer archetypes")
    print("3. Request more archetypes")
    print("4. Restart with modified rationale")

    choice = input("\nEnter your choice (1/2/3/4): ").strip()
    return choice

# ============================================
# STEP 6: SAVE RESULTS
# ============================================
def save_results(df, archetype_names):
    df.to_csv('archetype_results.csv', index=False)

    summary = {
        "total_buildings": len(df),
        "total_archetypes": len(archetype_names),
        "archetypes": []
    }

    for name in archetype_names:
        archetype_df = df[df['Archetype_Name'] == name]
        summary["archetypes"].append({
            "name": name,
            "count": len(archetype_df),
            "percentage": round(len(archetype_df)/len(df)*100, 1)
        })

    with open('archetype_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULTS SAVED ===")
    print("archetype_results.csv - Full dataset with archetype labels")
    print("archetype_results.json - Summary ready for Agent 2")
    print(f"\nTotal archetypes identified: {len(archetype_names)}")
    print(f"Total buildings classified: {len(df)}")

# ============================================
# MAIN PIPELINE
# ============================================
def main():
    print("\n" + "="*50)
    print("  IIDEAS LAB - AGENT 1: BUILDING ARCHETYPE ID")
    print("="*50)

    filepath = input("\nEnter path to your CSV file: ").strip()
    df = load_data(filepath)

    rationale = None
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

        # Run identification
        df_result, archetype_names, n_clusters = identify_archetypes(
            df.copy(), rationale, n_clusters
        )

        # Present results
        result_choice = present_results(df_result, archetype_names, rationale)

        if result_choice == "1":
            save_results(df_result, archetype_names)
            print("\nAgent 1 complete! Ready for Agent 2.")
            break

        elif result_choice == "2":
            n_clusters = max(2, n_clusters - 2)
            print(f"\nRerunning with {n_clusters} archetypes...")

        elif result_choice == "3":
            n_clusters = n_clusters + 2
            print(f"\nRerunning with {n_clusters} archetypes...")

        elif result_choice == "4":
            rationale = None
            n_clusters = None
            print("\nRestarting with modified rationale...")

if __name__ == "__main__":
    main()