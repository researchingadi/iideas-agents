import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE  = "archetype_results_full.csv"
OUTPUT_FILE = "ml_predictions.csv"
LABEL_COL   = "Archetype"

FEATURE_COLS = [
    "yearOfConstruction",
    "finishAreaSqft",
    "numOfBedrooms",
    "numOfFloors",
    "sizeSqft",
    "age",
    "grossFloorArea",
    "numOfRoomsTotal",
    "buildingHeightAverage",
    "buildingAssessedValueTotal",
]

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print("\n=== ML MODEL — BUILDING ARCHETYPE PREDICTION ===")
print(f"Loading {INPUT_FILE}...")

df = pd.read_csv(INPUT_FILE)
print(f"Total buildings: {len(df)}")

# ---------------------------------------------------------------------------
# Features and labels
# ---------------------------------------------------------------------------
available = [c for c in FEATURE_COLS if c in df.columns]
missing   = [c for c in FEATURE_COLS if c not in df.columns]
if missing:
    print(f"Note: {len(missing)} feature column(s) not found and skipped: {missing}")

X = pd.DataFrame()
for col in available:
    X[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
X = X.fillna(0).astype(float)

y = df[LABEL_COL].astype(str)

# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------
n_total = len(X)
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

print(f"Training set: {len(X_train)} (80%)")
print(f"Test set:     {len(X_test)} (20%)")

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
print("\nTraining Random Forest Classifier...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    min_samples_leaf=1,
)
rf.fit(X_train, y_train)
print("✅ Model trained successfully")

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------
y_pred   = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n=== RESULTS ===")
print(f"Accuracy: {accuracy * 100:.2f}%")

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
importance = sorted(
    zip(available, rf.feature_importances_),
    key=lambda x: x[1],
    reverse=True,
)

print("\nTop 5 Most Important Features:")
for rank, (feat, score) in enumerate(importance[:5], 1):
    print(f"  {rank}. {feat} ({score:.3f})")

# ---------------------------------------------------------------------------
# Predict on full dataset and save
# ---------------------------------------------------------------------------
all_preds = rf.predict(X)

out = df.copy()
out["ML_Predicted_Archetype"] = all_preds
out["ML_Correct"] = (out[LABEL_COL].astype(str) == all_preds)
out.to_csv(OUTPUT_FILE, index=False)

print(f"\nPredictions saved to: {OUTPUT_FILE}")
